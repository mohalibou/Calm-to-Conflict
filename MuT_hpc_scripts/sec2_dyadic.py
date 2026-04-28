from shared_utils import *

class DyadicSequenceDataset(Dataset):
    def __init__(self, df, text_dict, audio_dict, self_video_dict, partner_video_dict):
        self.samples = []
        for _, row in df.iterrows():
            s_id = row["sample_id"]
            if s_id in text_dict and s_id in audio_dict and s_id in self_video_dict:
                partner_v = partner_video_dict.get(s_id, torch.zeros_like(self_video_dict[s_id]))
                self.samples.append({
                    "label": int(row["label"]), "text": text_dict[s_id], "audio": audio_dict[s_id],
                    "video_self": self_video_dict[s_id], "video_partner": partner_v,
                })
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i = self.samples[idx]
        return (i["text"].clone().float(), i["audio"].clone().float(), i["video_self"].clone().float(), i["video_partner"].clone().float(), torch.tensor(i["label"], dtype=torch.long))

def mixup_batch(x_text, x_audio, x_vs, x_vp, y, alpha=0.2):
    batch_size, num_classes = y.size(0), 2
    if alpha <= 0:
        return x_text, x_audio, x_vs, x_vp, torch.zeros(batch_size, num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(batch_size, device=x_text.device)
    y_a = torch.zeros(batch_size, num_classes, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    y_b = torch.zeros(batch_size, num_classes, device=y.device).scatter_(1, y[idx].unsqueeze(1), 1.0)
    return (lam * x_text + (1 - lam) * x_text[idx], lam * x_audio + (1 - lam) * x_audio[idx], 
            lam * x_vs + (1 - lam) * x_vs[idx], lam * x_vp + (1 - lam) * x_vp[idx], lam * y_a + (1 - lam) * y_b)

class CalmToConflict_DyadicMulT(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, video_dim=92, shared_dim=128, num_heads=4, num_classes=2, dropout=0.2):
        super().__init__()
        self.shared_dim = shared_dim
        self.proj_text, self.proj_audio = nn.Conv1d(text_dim, shared_dim, 1), nn.Conv1d(audio_dim, shared_dim, 1)
        self.proj_vs, self.proj_vp = nn.Conv1d(video_dim, shared_dim, 1), nn.Conv1d(video_dim, shared_dim, 1)
        self.pos_encoder = PositionalEncoding(shared_dim, dropout)
        self.trans_T_with_A = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_T_with_VS = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_A_with_T = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_A_with_VS = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_VS_with_T = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_VS_with_A = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_VS_with_VP = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.trans_VP_with_VS = CrossModalAttentionBlock(shared_dim, num_heads, dropout)
        self.pool_T, self.pool_A = AttentionPool(shared_dim), AttentionPool(shared_dim)
        self.pool_VS, self.pool_VP = AttentionPool(shared_dim), AttentionPool(shared_dim)
        self.gate_T, self.gate_A = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        self.gate_VS, self.gate_VP = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 4, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x_text, x_audio, x_video_self, x_video_partner, use_text=True, use_audio=True, use_vs=True, use_vp=True):
        B, dev, D = x_text.size(0), x_text.device, self.shared_dim
        def _encode(x, proj, flag): return self.pos_encoder(proj(x.transpose(1, 2)).transpose(1, 2)) if flag else torch.zeros(B, x.size(1), D, device=dev)
        pT, pA = _encode(x_text, self.proj_text, use_text), _encode(x_audio, self.proj_audio, use_audio)
        pVS, pVP = _encode(x_video_self, self.proj_vs, use_vs), _encode(x_video_partner, self.proj_vp, use_vp)
        
        oT = self.trans_T_with_A(pT, pA) if use_text and use_audio else pT
        oT = self.trans_T_with_VS(oT, pVS) if use_text and use_vs else oT
        oA = self.trans_A_with_T(pA, pT) if use_audio and use_text else pA
        oA = self.trans_A_with_VS(oA, pVS) if use_audio and use_vs else oA
        oVS = self.trans_VS_with_T(pVS, pT) if use_vs and use_text else pVS
        oVS = self.trans_VS_with_A(oVS, pA) if use_vs and use_audio else oVS
        oVS = self.trans_VS_with_VP(oVS, pVP) if use_vs and use_vp else oVS
        oVP = self.trans_VP_with_VS(pVP, pVS) if use_vp and use_vs else pVP
        
        fused = torch.cat([
            torch.sigmoid(self.gate_T) * (self.pool_T(oT) if use_text else torch.zeros(B, D, device=dev)),
            torch.sigmoid(self.gate_A) * (self.pool_A(oA) if use_audio else torch.zeros(B, D, device=dev)),
            torch.sigmoid(self.gate_VS) * (self.pool_VS(oVS) if use_vs else torch.zeros(B, D, device=dev)),
            torch.sigmoid(self.gate_VP) * (self.pool_VP(oVP) if use_vp else torch.zeros(B, D, device=dev))
        ], dim=-1)
        return self.classifier(fused)

def train_eval(model, train_loader, test_loader, device, train_df, use_t, use_a, use_vs, use_vp, exp_name):
    model = model.to(device)
    weights = torch.tensor(1.0 / train_df["label"].value_counts().sort_index().values, dtype=torch.float32)
    weights = (weights / weights.sum()).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), 25 * len(train_loader))
    
    t_loss, v_loss, v_acc = [], [], []
    best_loss, best_wts, best_ep, patience_ctr = float("inf"), copy.deepcopy(model.state_dict()), 1, 0
    
    for epoch in range(25):
        model.train()
        run_loss = 0.0
        for bT, bA, bVS, bVP, by in train_loader:
            bT, bA, bVS, bVP, by = bT.to(device), bA.to(device), bVS.to(device), bVP.to(device), by.to(device)
            bT, bA, bVS, bVP, soft_y = mixup_batch(bT, bA, bVS, bVP, by, 0.2)
            soft_y = 0.95 * soft_y + 0.05 / 2
            
            logits = model(bT, bA, bVS, bVP, use_t, use_a, use_vs, use_vp)
            loss = -(soft_y * (F.log_softmax(logits, dim=-1) * weights.unsqueeze(0))).sum(dim=-1).mean()
            
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); run_loss += loss.item()
            
        t_loss.append(run_loss / len(train_loader))
        
        model.eval()
        run_v_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for bT, bA, bVS, bVP, by in test_loader:
                logits = model(bT.to(device), bA.to(device), bVS.to(device), bVP.to(device), use_t, use_a, use_vs, use_vp)
                y_onehot = torch.zeros_like(logits).scatter_(1, by.to(device).unsqueeze(1), 1.0)
                run_v_loss += -(y_onehot * (F.log_softmax(logits, dim=-1) * weights.unsqueeze(0))).sum(dim=-1).mean().item()
                preds.extend(torch.max(logits, 1)[1].cpu().numpy()); targets.extend(by.numpy())
                
        v_loss.append(run_v_loss / len(test_loader)); v_acc.append(accuracy_score(targets, preds))
        
        if v_loss[-1] < best_loss:
            best_loss, best_wts, best_ep, patience_ctr = v_loss[-1], copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            patience_ctr += 1
            if patience_ctr >= 10: break

    model.load_state_dict(best_wts)
    model.eval()
    f_preds, f_targets = [], []
    with torch.no_grad():
        for bT, bA, bVS, bVP, by in test_loader:
            logits = model(bT.to(device), bA.to(device), bVS.to(device), bVP.to(device), use_t, use_a, use_vs, use_vp)
            f_preds.extend(torch.max(logits, 1)[1].cpu().numpy()); f_targets.extend(by.numpy())
            
    acc, f1 = accuracy_score(f_targets, f_preds), f1_score(f_targets, f_preds, average="macro")
    save_training_plots(exp_name, t_loss, v_loss, v_acc, best_ep, f_preds, f_targets)
    return acc, f1, best_wts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Name of experiment to run")
    args = parser.parse_args()

    EXPERIMENTS = {
        "Full_Dyadic": (True, True, True, True), "No_Partner": (True, True, True, False),
        "SelfVideo_PartnerVideo": (False, False, True, True), "No_SelfVideo": (True, True, False, True),
        "No_Audio": (True, False, True, True), "No_Text": (False, True, True, True),
        "Only_PartnerVideo": (False, False, False, True), "Only_SelfVideo": (False, False, True, False),
        "Audio_SelfVideo": (False, True, True, False)
    }
    use_t, use_a, use_vs, use_vp = EXPERIMENTS[args.experiment]

    df = pd.read_csv(PATHS["csv"])
    text_dict, audio_dict = load_embedding("text"), load_embedding("audio")
    self_video_dict, partner_video_dict = load_embedding("video_self"), load_embedding("video_partner")
    train_df, test_df = group_split(df)

    train_loader = DataLoader(DyadicSequenceDataset(train_df, text_dict, audio_dict, self_video_dict, partner_video_dict), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(DyadicSequenceDataset(test_df, text_dict, audio_dict, self_video_dict, partner_video_dict), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_DIM, A_DIM, V_DIM = next(iter(text_dict.values())).shape[-1], next(iter(audio_dict.values())).shape[-1], next(iter(self_video_dict.values())).shape[-1]
    
    acc_runs, f1_runs, best_f1, best_state = [], [], -1.0, None
    for seed in RANDOM_SEEDS:
        set_seed(seed)
        model = CalmToConflict_DyadicMulT(text_dim=T_DIM, audio_dim=A_DIM, video_dim=V_DIM).to(device)
        acc, f1, wts = train_eval(model, train_loader, test_loader, device, train_df, use_t, use_a, use_vs, use_vp, f"{args.experiment}_s{seed}")
        acc_runs.append(acc); f1_runs.append(f1)
        if f1 > best_f1: best_f1, best_state = f1, wts

    torch.save(best_state, os.path.join(PATHS["models"], f"{args.experiment}_best.pt"))
    with open(os.path.join(PATHS["results"], f"dyadic_{args.experiment}.json"), "w") as f:
        json.dump({args.experiment: aggregate_seeds(acc_runs, f1_runs)}, f, indent=2)