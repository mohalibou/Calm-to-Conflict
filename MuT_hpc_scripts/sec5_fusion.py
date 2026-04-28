from shared_utils import *

class DyadicAudioVisualDataset(Dataset):
    def __init__(self, df, audio_dict, self_video_dict, partner_video_dict):
        self.samples = []
        for _, row in df.iterrows():
            s_id = row["sample_id"]
            if s_id in audio_dict and s_id in self_video_dict:
                vp = partner_video_dict.get(s_id, torch.zeros_like(self_video_dict[s_id]))
                self.samples.append({"label": int(row["label"]), "audio": audio_dict[s_id], "video_self": self_video_dict[s_id], "video_partner": vp})
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i = self.samples[idx]
        return i["audio"].clone().float(), i["video_self"].clone().float(), i["video_partner"].clone().float(), torch.tensor(i["label"], dtype=torch.long)

def mixup_fusion(streams, y, alpha=0.2):
    B, NC = y.size(0), 2
    if alpha <= 0: return streams, torch.zeros(B, NC, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(B, device=streams[0].device)
    mixed = [lam * s + (1 - lam) * s[idx] for s in streams]
    y_a = torch.zeros(B, NC, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    y_b = torch.zeros(B, NC, device=y.device).scatter_(1, y[idx].unsqueeze(1), 1.0)
    return mixed, lam * y_a + (1 - lam) * y_b

class Audio_DyadicFAUBody_MulT(nn.Module):
    def __init__(self, audio_dim=768, fau_dim=24, body_dim=63, shared_dim=64, num_heads=4):
        super().__init__()
        self.proj_audio = nn.Conv1d(audio_dim, shared_dim, 1)
        self.proj_vs_fau, self.proj_vp_fau = nn.Conv1d(fau_dim, shared_dim, 1), nn.Conv1d(fau_dim, shared_dim, 1)
        self.proj_vs_bod, self.proj_vp_bod = nn.Conv1d(body_dim, shared_dim, 1), nn.Conv1d(body_dim, shared_dim, 1)
        
        self.pos_encoder = PositionalEncoding(shared_dim, 0.2)
        
        self.trans_A_vsFAU, self.trans_A_vsBod = CrossModalAttentionBlock(shared_dim, num_heads), CrossModalAttentionBlock(shared_dim, num_heads)
        self.trans_vsFAU_A, self.trans_vsFAU_Bod = CrossModalAttentionBlock(shared_dim, num_heads), CrossModalAttentionBlock(shared_dim, num_heads)
        self.trans_vsBod_A, self.trans_vsBod_FAU = CrossModalAttentionBlock(shared_dim, num_heads), CrossModalAttentionBlock(shared_dim, num_heads)
        
        self.trans_vsFAU_vpFAU, self.trans_vpFAU_vsFAU = CrossModalAttentionBlock(shared_dim, num_heads), CrossModalAttentionBlock(shared_dim, num_heads)
        self.trans_vsBod_vpBod, self.trans_vpBod_vsBod = CrossModalAttentionBlock(shared_dim, num_heads), CrossModalAttentionBlock(shared_dim, num_heads)
        
        self.pool_A, self.pool_vs_FAU, self.pool_vs_Bod = AttentionPool(shared_dim), AttentionPool(shared_dim), AttentionPool(shared_dim)
        self.pool_vp_FAU, self.pool_vp_Bod = AttentionPool(shared_dim), AttentionPool(shared_dim)
        
        self.gate_A, self.gate_vs_FAU, self.gate_vs_Bod = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        self.gate_vp_FAU, self.gate_vp_Bod = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        
        self.classifier = nn.Sequential(nn.Linear(shared_dim * 5, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4), nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x_audio, x_vs_fau, x_vs_bod, x_vp_fau, x_vp_bod):
        pA = self.pos_encoder(self.proj_audio(x_audio.transpose(1, 2)).transpose(1, 2))
        pVSF, pVSB = self.pos_encoder(self.proj_vs_fau(x_vs_fau.transpose(1, 2)).transpose(1, 2)), self.pos_encoder(self.proj_vs_bod(x_vs_bod.transpose(1, 2)).transpose(1, 2))
        pVPF, pVPB = self.pos_encoder(self.proj_vp_fau(x_vp_fau.transpose(1, 2)).transpose(1, 2)), self.pos_encoder(self.proj_vp_bod(x_vp_bod.transpose(1, 2)).transpose(1, 2))
        
        oA = self.trans_A_vsBod(self.trans_A_vsFAU(pA, pVSF), pVSB)
        oVSF = self.trans_vsFAU_vpFAU(self.trans_vsFAU_Bod(self.trans_vsFAU_A(pVSF, pA), pVSB), pVPF)
        oVSB = self.trans_vsBod_vpBod(self.trans_vsBod_FAU(self.trans_vsBod_A(pVSB, pA), pVSF), pVPB)
        
        oVPF, oVPB = self.trans_vpFAU_vsFAU(pVPF, pVSF), self.trans_vpBod_vsBod(pVPB, pVSB)
        
        fused = torch.cat([
            torch.sigmoid(self.gate_A) * self.pool_A(oA), 
            torch.sigmoid(self.gate_vs_FAU) * self.pool_vs_FAU(oVSF), torch.sigmoid(self.gate_vs_Bod) * self.pool_vs_Bod(oVSB),
            torch.sigmoid(self.gate_vp_FAU) * self.pool_vp_FAU(oVPF), torch.sigmoid(self.gate_vp_Bod) * self.pool_vp_Bod(oVPB)
        ], dim=-1)
        return self.classifier(fused)

def train_eval_fusion(model, train_loader, test_loader, device, train_df, fau_slice, body_slice, exp_name):
    model = model.to(device)
    weights = torch.tensor(1.0 / train_df["label"].value_counts().sort_index().values, dtype=torch.float32).to(device)
    weights = weights / weights.sum()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), 25 * len(train_loader))
    
    t_loss, v_loss, v_acc = [], [], []
    best_loss, best_wts, best_ep, patience_ctr = float("inf"), copy.deepcopy(model.state_dict()), 1, 0
    
    for epoch in range(25):
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            bA, bVS, bVP, by = [t.to(device) for t in batch]
            bVS_FAU, bVS_Bod = bVS[:, :, fau_slice[0]:fau_slice[1]], bVS[:, :, body_slice[0]:body_slice[1]]
            bVP_FAU, bVP_Bod = bVP[:, :, fau_slice[0]:fau_slice[1]], bVP[:, :, body_slice[0]:body_slice[1]]
            
            mixed, soft_y = mixup_fusion([bA, bVS_FAU, bVS_Bod, bVP_FAU, bVP_Bod], by, 0.2)
            logits = model(*mixed)
                
            soft_y = 0.95 * soft_y + 0.05 / 2
            loss = -(soft_y * (F.log_softmax(logits, dim=-1) * weights.unsqueeze(0))).sum(dim=-1).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run_loss += loss.item()
            
        t_loss.append(run_loss / len(train_loader))
        
        model.eval()
        run_v_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                bA, bVS, bVP, by = [t.to(device) for t in batch]
                bVS_FAU, bVS_Bod = bVS[:, :, fau_slice[0]:fau_slice[1]], bVS[:, :, body_slice[0]:body_slice[1]]
                bVP_FAU, bVP_Bod = bVP[:, :, fau_slice[0]:fau_slice[1]], bVP[:, :, body_slice[0]:body_slice[1]]
                
                logits = model(bA, bVS_FAU, bVS_Bod, bVP_FAU, bVP_Bod)
                    
                y_onehot = torch.zeros_like(logits).scatter_(1, by.unsqueeze(1), 1.0)
                run_v_loss += -(y_onehot * (F.log_softmax(logits, dim=-1) * weights.unsqueeze(0))).sum(dim=-1).mean().item()
                preds.extend(torch.max(logits, 1)[1].cpu().numpy())
                targets.extend(by.cpu().numpy())
                
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
        for batch in test_loader:
            bA, bVS, bVP, by = [t.to(device) for t in batch]
            bVS_FAU, bVS_Bod = bVS[:, :, fau_slice[0]:fau_slice[1]], bVS[:, :, body_slice[0]:body_slice[1]]
            bVP_FAU, bVP_Bod = bVP[:, :, fau_slice[0]:fau_slice[1]], bVP[:, :, body_slice[0]:body_slice[1]]
            
            logits = model(bA, bVS_FAU, bVS_Bod, bVP_FAU, bVP_Bod)
            f_preds.extend(torch.max(logits, 1)[1].cpu().numpy())
            f_targets.extend(by.cpu().numpy())
            
    acc, f1 = accuracy_score(f_targets, f_preds), f1_score(f_targets, f_preds, average="macro")
    save_training_plots(exp_name, t_loss, v_loss, v_acc, best_ep, f_preds, f_targets)
    return acc, f1, best_wts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, choices=["Audio_DyadicFAUBody"], help="Name of 5-stream fusion experiment")
    args = parser.parse_args()

    with open(EMBEDDINGS["video_feat_dims"]) as f:
        meta = json.load(f)
        fau_slice = meta["fau_slice"]
        body_slice = meta["body_slice"]

    df = pd.read_csv(PATHS["csv"])
    audio_dict, self_video_dict = load_embedding("audio"), load_embedding("video_self")
    partner_video_dict = load_embedding("video_partner")
    train_df, test_df = group_split(df)

    train_loader = DataLoader(DyadicAudioVisualDataset(train_df, audio_dict, self_video_dict, partner_video_dict), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(DyadicAudioVisualDataset(test_df, audio_dict, self_video_dict, partner_video_dict), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_DIM, FAU_DIM, BOD_DIM = next(iter(audio_dict.values())).shape[-1], fau_slice[1] - fau_slice[0], body_slice[1] - body_slice[0]

    acc_runs, f1_runs, best_f1, best_state = [], [], -1.0, None
    for seed in RANDOM_SEEDS:
        set_seed(seed)
        model = Audio_DyadicFAUBody_MulT(audio_dim=A_DIM, fau_dim=FAU_DIM, body_dim=BOD_DIM).to(device)
        acc, f1, wts = train_eval_fusion(model, train_loader, test_loader, device, train_df, fau_slice, body_slice, f"{args.experiment}_s{seed}")
        acc_runs.append(acc); f1_runs.append(f1)
        if f1 > best_f1: best_f1, best_state = f1, wts

    torch.save(best_state, os.path.join(PATHS["models"], f"{args.experiment}_best.pt"))
    with open(os.path.join(PATHS["results"], f"fusion_{args.experiment}.json"), "w") as f:
        json.dump({args.experiment: aggregate_seeds(acc_runs, f1_runs)}, f, indent=2)