from shared_utils import *

def load_feature_dims(json_path="embeddings/video_feature_dims.json"):
    with open(json_path) as f:
        meta = json.load(f)
    return {
        "fau":  tuple(meta["fau_slice"]), "head": tuple(meta["head_slice"]),
        "gaze": tuple(meta["gaze_slice"]), "body": tuple(meta["body_slice"]),
    }

class VisualAblationDataset(Dataset):
    def __init__(self, df, audio_dict, self_video_dict, partner_video_dict, feature_slices):
        self.samples = []
        fs = feature_slices
        for _, row in df.iterrows():
            sid = row["sample_id"]
            if sid in self_video_dict:
                vs = self_video_dict[sid]
                vp = partner_video_dict.get(sid, torch.zeros_like(vs))
                self.samples.append({
                    "label": int(row["label"]),
                    "vs_fau": vs[:, fs["fau"][0]:fs["fau"][1]], "vs_head": vs[:, fs["head"][0]:fs["head"][1]],
                    "vs_gaze": vs[:, fs["gaze"][0]:fs["gaze"][1]], "vs_body": vs[:, fs["body"][0]:fs["body"][1]],
                    "vp_fau": vp[:, fs["fau"][0]:fs["fau"][1]], "vp_head": vp[:, fs["head"][0]:fs["head"][1]],
                    "vp_gaze": vp[:, fs["gaze"][0]:fs["gaze"][1]], "vp_body": vp[:, fs["body"][0]:fs["body"][1]],
                })
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i = self.samples[idx]
        return (i["vs_fau"].clone().float(), i["vs_head"].clone().float(), i["vs_gaze"].clone().float(), i["vs_body"].clone().float(),
                i["vp_fau"].clone().float(), i["vp_head"].clone().float(), i["vp_gaze"].clone().float(), i["vp_body"].clone().float(),
                torch.tensor(i["label"], dtype=torch.long))

def mixup_batch_vis(streams, y, alpha=0.2):
    B, NC = y.size(0), 2
    if alpha <= 0: return streams, torch.zeros(B, NC, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(B, device=streams[0].device)
    mixed = [lam * s + (1 - lam) * s[idx] for s in streams]
    y_a = torch.zeros(B, NC, device=y.device).scatter_(1, y.unsqueeze(1), 1.0)
    y_b = torch.zeros(B, NC, device=y.device).scatter_(1, y[idx].unsqueeze(1), 1.0)
    return mixed, lam * y_a + (1 - lam) * y_b

class VisualDyadicMulT(nn.Module):
    STREAM_KEYS = ["fau", "head", "gaze", "body"]
    def __init__(self, feature_dims, shared_dim=64, num_heads=4, num_classes=2, dropout=0.2):
        super().__init__()
        self.shared_dim, self.keys = shared_dim, self.STREAM_KEYS
        self.proj = nn.ModuleDict({k: nn.Conv1d(feature_dims[k], shared_dim, 1) for k in self.keys})
        self.pos_encoder = PositionalEncoding(shared_dim, dropout)
        self.intra_attn = nn.ModuleDict({f"{q}_{kv}": CrossModalAttentionBlock(shared_dim, num_heads, dropout) for q in self.keys for kv in self.keys if q != kv})
        self.cross_self_to_partner = nn.ModuleDict({k: CrossModalAttentionBlock(shared_dim, num_heads, dropout) for k in self.keys})
        self.cross_partner_to_self = nn.ModuleDict({k: CrossModalAttentionBlock(shared_dim, num_heads, dropout) for k in self.keys})
        self.pool_vs = nn.ModuleDict({k: AttentionPool(shared_dim) for k in self.keys})
        self.pool_vp = nn.ModuleDict({k: AttentionPool(shared_dim) for k in self.keys})
        self.gate_vs = nn.ParameterDict({k: nn.Parameter(torch.ones(1)) for k in self.keys})
        self.gate_vp = nn.ParameterDict({k: nn.Parameter(torch.ones(1)) for k in self.keys})
        self._max_fused_dim = shared_dim * len(self.keys) * 2
        self.classifier_hidden = nn.Sequential(nn.Linear(self._max_fused_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3))
        self.classifier_head = nn.Linear(128, num_classes)
        self.input_proj = nn.Linear(self._max_fused_dim, self._max_fused_dim)

    def forward(self, sub_streams, active_vs, active_vp):
        B, dev, D = next(iter(sub_streams.values())).size(0), next(iter(sub_streams.values())).device, self.shared_dim
        def _encode(x, key): return self.pos_encoder(self.proj[key](x.transpose(1, 2)).transpose(1, 2))
        enc_vs = {k: _encode(sub_streams[f"vs_{k}"], k) for k in active_vs}
        enc_vp = {k: _encode(sub_streams[f"vp_{k}"], k) for k in active_vp}
        out_vs = {k: enc_vs[k].clone() for k in active_vs}
        for q in active_vs:
            for kv in active_vs:
                if q != kv: out_vs[q] = self.intra_attn[f"{q}_{kv}"](out_vs[q], enc_vs[kv])
        out_vp = {k: enc_vp[k].clone() for k in active_vp}
        for k in active_vs:
            if k in active_vp: out_vs[k] = self.cross_self_to_partner[k](out_vs[k], enc_vp[k])
        for k in active_vp:
            if k in active_vs: out_vp[k] = self.cross_partner_to_self[k](out_vp[k], enc_vs[k])
        
        pooled_parts = []
        for k in self.keys:
            pooled_parts.append(torch.sigmoid(self.gate_vs[k]) * self.pool_vs[k](out_vs[k]) if k in active_vs else torch.zeros(B, D, device=dev))
            pooled_parts.append(torch.sigmoid(self.gate_vp[k]) * self.pool_vp[k](out_vp[k]) if k in active_vp else torch.zeros(B, D, device=dev))
        
        return self.classifier_head(self.classifier_hidden(self.input_proj(torch.cat(pooled_parts, dim=-1))))

def train_eval_vis(model, train_loader, test_loader, device, train_df, active_vs, active_vp, exp_name):
    model = model.to(device)
    weights = torch.tensor(1.0 / train_df["label"].value_counts().sort_index().values, dtype=torch.float32)
    weights = (weights / weights.sum()).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader), 25 * len(train_loader))
    stream_keys = ["vs_fau", "vs_head", "vs_gaze", "vs_body", "vp_fau", "vp_head", "vp_gaze", "vp_body"]
    
    t_loss, v_loss, v_acc = [], [], []
    best_loss, best_wts, best_ep, patience_ctr = float("inf"), copy.deepcopy(model.state_dict()), 1, 0
    
    for epoch in range(25):
        model.train()
        run_loss = 0.0
        for batch in train_loader:
            *stream_tensors, by = batch
            stream_tensors, by = [t.to(device) for t in stream_tensors], by.to(device)
            mixed, soft_y = mixup_batch_vis(stream_tensors, by, 0.2)
            soft_y = 0.95 * soft_y + 0.05 / 2
            
            logits = model(dict(zip(stream_keys, mixed)), active_vs, active_vp)
            loss = -(soft_y * (F.log_softmax(logits, dim=-1) * weights.unsqueeze(0))).sum(dim=-1).mean()
            
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); run_loss += loss.item()
        t_loss.append(run_loss / len(train_loader))
        
        model.eval()
        run_v_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                *stream_tensors, by = batch
                stream_tensors, by = [t.to(device) for t in stream_tensors], by.to(device)
                logits = model(dict(zip(stream_keys, stream_tensors)), active_vs, active_vp)
                y_onehot = torch.zeros_like(logits).scatter_(1, by.unsqueeze(1), 1.0)
                run_v_loss += -(y_onehot * (F.log_softmax(logits, dim=-1) * weights.unsqueeze(0))).sum(dim=-1).mean().item()
                preds.extend(torch.max(logits, 1)[1].cpu().numpy()); targets.extend(by.cpu().numpy())
        v_loss.append(run_v_loss / len(test_loader)); v_acc.append(accuracy_score(targets, preds))
        
        if v_loss[-1] < best_loss:
            best_loss, best_wts, best_ep, patience_ctr = v_loss[-1], copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            patience_ctr += 1
            if patience_ctr >= 6: break

    model.load_state_dict(best_wts)
    model.eval()
    f_preds, f_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            *stream_tensors, by = batch
            logits = model(dict(zip(stream_keys, [t.to(device) for t in stream_tensors])), active_vs, active_vp)
            f_preds.extend(torch.max(logits, 1)[1].cpu().numpy()); f_targets.extend(by.numpy())
            
    acc, f1 = accuracy_score(f_targets, f_preds), f1_score(f_targets, f_preds, average="macro")
    save_training_plots(exp_name, t_loss, v_loss, v_acc, best_ep, f_preds, f_targets)
    return acc, f1, best_wts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Name of visual experiment")
    args = parser.parse_args()

    VIS_EXPERIMENTS = {
        "Full_Visual_Dyadic": ({"fau","head","gaze","body"}, {"fau","head","gaze","body"}), "FAU_Only": ({"fau"}, {"fau"}),
        "Head_Only": ({"head"}, {"head"}), "Gaze_Only": ({"gaze"}, {"gaze"}), "Body_Only": ({"body"}, {"body"}),
        "FAU_Head": ({"fau","head"}, {"fau","head"}), "FAU_Gaze": ({"fau","gaze"}, {"fau","gaze"}),
        "FAU_Body": ({"fau","body"}, {"fau","body"}), "Face_Only": ({"fau","head","gaze"}, {"fau","head","gaze"}),
        "Body_Speaker_FAU_Partner": ({"body"}, {"fau"})
    }
    active_vs, active_vp = VIS_EXPERIMENTS[args.experiment]
    dims = load_feature_dims()

    df = pd.read_csv(PATHS["csv"])
    audio_dict, self_video_dict, partner_video_dict = load_embedding("audio"), load_embedding("video_self"), load_embedding("video_partner")
    train_df, test_df = group_split(df)

    train_loader = DataLoader(VisualAblationDataset(train_df, audio_dict, self_video_dict, partner_video_dict, dims), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(VisualAblationDataset(test_df, audio_dict, self_video_dict, partner_video_dict, dims), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dims_dict = {k: dims[f"{k}"][1] - dims[f"{k}"][0] for k in ["fau", "head", "gaze", "body"]}

    acc_runs, f1_runs, best_f1, best_state = [], [], -1.0, None
    for seed in RANDOM_SEEDS:
        set_seed(seed)
        model = VisualDyadicMulT(feature_dims=feature_dims_dict, shared_dim=64, num_heads=4).to(device)
        acc, f1, wts = train_eval_vis(model, train_loader, test_loader, device, train_df, active_vs, active_vp, f"{args.experiment}_s{seed}")
        acc_runs.append(acc); f1_runs.append(f1)
        if f1 > best_f1: best_f1, best_state = f1, wts

    torch.save(best_state, os.path.join(PATHS["models"], f"{args.experiment}_best.pt"))
    with open(os.path.join(PATHS["results"], f"vis_{args.experiment}.json"), "w") as f:
        json.dump({args.experiment: aggregate_seeds(acc_runs, f1_runs)}, f, indent=2)