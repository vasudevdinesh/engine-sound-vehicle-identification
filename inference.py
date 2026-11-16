import sys, os, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import librosa
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")

TARGET_SR = 32000
WIN_LEN = int(TARGET_SR * 3.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "master_5head_engine_model (3).pth"

CAR_MODEL_DICT = {
    "K10": ["Maruti Alto K10", "S-Presso", "Celerio", "WagonR 1.0"],
    "K12_1.2_NA": ["Maruti Swift", "Baleno", "Dzire", "Ignis", "Toyota Glanza"],
    "K15": ["Maruti Brezza", "Ertiga", "XL6", "Grand Vitara", "Toyota Hyryder"],
    "i-VTEC_1.2": ["Honda Amaze", "Jazz", "WR-V", "Brio"],
    "L15Z_1.5_iVTEC": ["Honda City 1.5"],
    "Energy_1.0": ["Renault Triber", "Kiger NA", "Kwid 1.0", "Nissan Magnite NA"],
    "Revotron_1.2_NA": ["Tata Punch", "Tiago", "Tigor", "Altroz NA"],
    "Kappa_1.0_Turbo": ["Hyundai Venue", "i20 N Line", "Kia Sonet"],
    "EA211_1.0_TSI": ["VW Taigun", "Virtus", "Skoda Kushaq", "Slavia", "Kylaq"],
    "mHawk_2.2_Diesel": ["Mahindra Thar", "Scorpio", "XUV700"],
    "D13A_1.3_DDiS": ["Maruti Swift Diesel (Pre-2020)", "Brezza Diesel (Pre-2020)"],
    "1ND_1.4_D4D": ["Toyota Corolla Altis Diesel", "Etios Diesel"]
}

# 1. DSP SANITIZATION
def bandpass_15_10k(y, sr=TARGET_SR):
    nyq = 0.5 * sr
    sos = butter(4, [15.0 / nyq, min(10000.0 / nyq, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, y)

def sanitize_recording(y, sr=TARGET_SR):
    y = bandpass_15_10k(y, sr)
    chunk_len = int(0.5 * sr)
    num_chunks = len(y) // chunk_len
    if num_chunks == 0:
        return np.array([])
        
    chunks = [y[i * chunk_len : (i + 1) * chunk_len] for i in range(num_chunks)]
    chunk_rms = np.array([np.sqrt(np.mean(c**2)) for c in chunks])
    chunk_flatness = np.array([float(np.mean(librosa.feature.spectral_flatness(y=c))) for c in chunks])
    chunk_centroid = np.array([float(np.mean(librosa.feature.spectral_centroid(y=c, sr=sr))) for c in chunks])
    
    active_mask = chunk_rms > 1e-4
    if np.sum(active_mask) == 0:
        return np.array([])
        
    median_rms = np.median(chunk_rms[active_mask])
    median_centroid = np.median(chunk_centroid[active_mask])
    
    clean_chunks = []
    fade_len = int(0.01 * sr)
    window_taper = np.ones(chunk_len, dtype=np.float32)
    window_taper[:fade_len] = np.linspace(0.0, 1.0, fade_len)
    window_taper[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    
    for c, r, flat, cent in zip(chunks, chunk_rms, chunk_flatness, chunk_centroid):
        if r < 0.30 * median_rms or r > 1.30 * median_rms or cent > 1.20 * median_centroid or flat > 0.35:
            continue
        clean_chunks.append(c * window_taper)
        
    if len(clean_chunks) < 6:
        return np.array([])
        
    y_clean = np.concatenate(clean_chunks)
    final_rms = np.sqrt(np.mean(y_clean**2)) + 1e-8
    return np.clip(y_clean * (0.08 / final_rms), -0.99, 0.99)

# 2. MODEL ARCHITECTURE
class AttentionPool2d(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_features, in_features // 4, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(in_features // 4, 1, kernel_size=1)
        )
    def forward(self, x):
        weights = F.softmax(self.attn(x).view(x.size(0), 1, -1), dim=-1)
        x_flat = x.view(x.size(0), x.size(1), -1)
        return torch.bmm(x_flat, weights.transpose(1, 2)).squeeze(-1)

class MultiHeadEngineResNetPro(nn.Module):
    def __init__(self, num_families):
        super().__init__()
        base = models.resnet34()
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.attn_pool = AttentionPool2d(base.fc.in_features)
        self.bottleneck = nn.Sequential(
            nn.Linear(base.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.35)
        )
        self.head_cyl = nn.Linear(256, 2)
        self.head_fuel = nn.Linear(256, 2)
        self.head_turbo = nn.Linear(256, 2)
        self.head_state = nn.Linear(256, 2)
        self.head_family = nn.Linear(256 + 6, num_families)

    def forward(self, x):
        feat = self.bottleneck(self.attn_pool(self.feature_extractor(x)))
        out_cyl = self.head_cyl(feat)
        out_fuel = self.head_fuel(feat)
        out_turbo = self.head_turbo(feat)
        fam_input = torch.cat([feat, out_cyl.detach(), out_fuel.detach(), out_turbo.detach()], dim=1)
        return {
            "cyl": out_cyl,
            "fuel": out_fuel,
            "turbo": out_turbo,
            "family": self.head_family(fam_input)
        }

# 3. LOAD WEIGHTS
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found.")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
valid_fams = checkpoint["valid_fams"]
family_spec_lookup = checkpoint["family_spec_lookup"]
inv_fam_map = {i: f for i, f in enumerate(valid_fams)}

models_eval = {}
for k in ["cyl", "ft", "fam"]:
    m = MultiHeadEngineResNetPro(num_families=len(valid_fams)).to(DEVICE)
    m.load_state_dict(checkpoint[f"{k}_state_dict"])
    m.eval()
    models_eval[k] = m

# 4. INFERENCE PIPELINE
def predict_engine(audio_path):
    if not os.path.exists(audio_path):
        return f"Error: File '{audio_path}' not found."

    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    y_clean = sanitize_recording(y)
    
    if len(y_clean) == 0:
        return "Audio rejected by DSP sanitizer (too much wind, talking, or silence)."

    num_wins = min(len(y_clean) // WIN_LEN, 4)
    if num_wins == 0:
        y_clean = np.pad(y_clean, (0, max(0, WIN_LEN - len(y_clean))), mode="wrap")[:WIN_LEN]
        num_wins = 1

    probs = {"cyl": [], "fuel": [], "turbo": [], "family": []}
    
    for w_idx in range(num_wins):
        y_win = y_clean[w_idx * WIN_LEN : (w_idx + 1) * WIN_LEN]
        mel = librosa.feature.melspectrogram(
            y=y_win, sr=TARGET_SR, n_fft=2048, hop_length=512,
            n_mels=128, fmin=15.0, fmax=14000.0, power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        
        spec_tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(DEVICE)
        
        with torch.no_grad():
            out_c = models_eval["cyl"](spec_tensor)
            out_ft = models_eval["ft"](spec_tensor)
            out_fm = models_eval["fam"](spec_tensor)
            
            probs["cyl"].append(F.softmax(out_c["cyl"], dim=1).cpu().numpy()[0])
            probs["fuel"].append(F.softmax(out_ft["fuel"], dim=1).cpu().numpy()[0])
            probs["turbo"].append(F.softmax(out_ft["turbo"], dim=1).cpu().numpy()[0])
            probs["family"].append(F.softmax(out_fm["family"], dim=1).cpu().numpy()[0])

    p_cyl = np.mean(probs["cyl"], axis=0)
    p_fuel = np.mean(probs["fuel"], axis=0)
    p_turbo = np.mean(probs["turbo"], axis=0)
    p_fam = np.mean(probs["family"], axis=0)

    # Bayesian Soft Fusion with Reliability Exponents:
    # Fuel (100% acc) & Turbo (100% acc) get strict weight (2.0)
    # Cylinder (84.6% acc) gets moderate soft weight (1.2) so 49/51 splits don't zero out families
    # Family (46.2% acc) gets base weight (1.0)
    guided_fam = np.zeros(len(valid_fams), dtype=np.float32)
    for f_idx, (req_c, req_f, req_t) in family_spec_lookup.items():
        phys_prior = (p_cyl[req_c] ** 1.2) * (p_fuel[req_f] ** 2.0) * (p_turbo[req_t] ** 2.0)
        guided_fam[f_idx] = (p_fam[f_idx] ** 1.0) * phys_prior

    # Normalize into a posterior probability distribution (sums to 100%)
    guided_fam = guided_fam / (np.sum(guided_fam) + 1e-8)
    top2_idx = np.argsort(guided_fam)[-2:][::-1]

    # Sync displayed physical specs with the winning posterior family
    win_c, win_f, win_t = family_spec_lookup[top2_idx[0]]
    cyl_str = f"{'4-Cylinder' if win_c == 1 else '3-Cylinder'} ({p_cyl[win_c]*100:.1f}%)"
    fuel_str = f"{'Diesel' if win_f == 1 else 'Petrol'} ({p_fuel[win_f]*100:.1f}%)"
    turbo_str = f"{'Turbocharged' if win_t == 1 else 'Naturally Aspirated'} ({p_turbo[win_t]*100:.1f}%)"

    top1_fam = inv_fam_map.get(top2_idx[0])
    top2_fam = inv_fam_map.get(top2_idx[1])
    
    top1_cars = ", ".join(CAR_MODEL_DICT.get(top1_fam, ["Unknown Models"]))
    top2_cars = ", ".join(CAR_MODEL_DICT.get(top2_fam, ["Unknown Models"]))

    res = f"\nAnalyzed: {os.path.basename(audio_path)}\n"
    res += "-" * 55 + "\n"
    res += f"Physical Specs : {cyl_str} | {fuel_str} | {turbo_str}\n"
    res += "-" * 55 + "\n"
    res += f"Top Match      : {top1_fam} (Conf: {guided_fam[top2_idx[0]]*100:.1f}%)\n"
    res += f"   Likely Cars : {top1_cars}\n\n"
    res += f"Runner Up      : {top2_fam} (Conf: {guided_fam[top2_idx[1]]*100:.1f}%)\n"
    res += f"   Likely Cars : {top2_cars}\n"
    res += "-" * 55 + "\n"
    return res

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio_file>")
    else:
        print(predict_engine(sys.argv[1]))