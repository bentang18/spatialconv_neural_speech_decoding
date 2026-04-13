#!/usr/bin/env python3
"""Test CPU vs CUDA numerical divergence for our model pipeline."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Build a model mimicking our pipeline
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_spatial = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.ln = nn.LayerNorm(256)
        self.conv_temp = nn.Conv1d(256, 32, 10, stride=10)
        self.gru = nn.GRU(32, 32, 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(64, 9)

    def forward(self, x):
        B, H, W, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        x = F.relu(self.conv_spatial(x))
        x = self.pool(x)
        x = x.reshape(B, T, -1)
        x = self.ln(x).permute(0, 2, 1)
        x = F.gelu(self.conv_temp(x)).permute(0, 2, 1)
        h, _ = self.gru(x)
        return self.head(h.mean(dim=1))


# Create model and input
torch.manual_seed(42)
model = TestModel()
x = torch.randn(4, 8, 16, 201)

# === Test 1: AdaptiveAvgPool2d CPU vs CUDA ===
conv = nn.Conv2d(1, 8, 3, padding=1)
pool = nn.AdaptiveAvgPool2d((4, 8))
small_x = torch.randn(4, 1, 8, 16)

out_cpu = pool(F.relu(conv(small_x)))

conv_c = nn.Conv2d(1, 8, 3, padding=1).cuda()
conv_c.load_state_dict(conv.state_dict())
pool_c = nn.AdaptiveAvgPool2d((4, 8)).cuda()
out_cuda = pool_c(F.relu(conv_c(small_x.cuda()))).cpu()

diff = (out_cpu - out_cuda).abs()
print(f"AdaptiveAvgPool2d CPU vs CUDA: max={diff.max().item():.2e}, mean={diff.mean().item():.2e}")

# === Test 2: GRU CPU vs CUDA ===
torch.manual_seed(42)
gru = nn.GRU(32, 32, num_layers=2, batch_first=True, bidirectional=True)
inp = torch.randn(4, 20, 32)

out_gru_cpu, _ = gru(inp)

gru_c = nn.GRU(32, 32, num_layers=2, batch_first=True, bidirectional=True).cuda()
gru_c.load_state_dict(gru.state_dict())
out_gru_cuda, _ = gru_c(inp.cuda())

diff_gru = (out_gru_cpu - out_gru_cuda.cpu()).abs()
print(f"GRU CPU vs CUDA: max={diff_gru.max().item():.2e}, mean={diff_gru.mean().item():.2e}")

# === Test 3: CUDA GRU self-determinism ===
out1, _ = gru_c(inp.cuda())
out2, _ = gru_c(inp.cuda())
diff_det = (out1 - out2).abs()
print(f"CUDA GRU self-consistency: max={diff_det.max().item():.2e} (0 = deterministic)")

# === Test 4: Full model CPU vs CUDA ===
torch.manual_seed(42)
model = TestModel()
x = torch.randn(4, 8, 16, 201)

out_cpu = model(x)

model_c = TestModel().cuda()
model_c.load_state_dict(model.state_dict())
out_cuda = model_c(x.cuda()).cpu()

diff_full = (out_cpu - out_cuda).abs()
print(f"\nFull model CPU vs CUDA: max={diff_full.max().item():.2e}, mean={diff_full.mean().item():.2e}")
print(f"CPU  logits[0,:3]: {[f'{v:.6f}' for v in out_cpu[0,:3].tolist()]}")
print(f"CUDA logits[0,:3]: {[f'{v:.6f}' for v in out_cuda[0,:3].tolist()]}")

# === Test 5: Gradient step divergence ===
print("\n=== Training divergence after N steps ===")
import copy

torch.manual_seed(42)
model_cpu = TestModel()
model_cuda = copy.deepcopy(model_cpu).cuda()
opt_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=1e-3)
opt_cuda = torch.optim.AdamW(model_cuda.parameters(), lr=1e-3)

torch.manual_seed(42)
for step in range(20):
    x = torch.randn(4, 8, 16, 201)
    target = torch.randint(0, 9, (4,))

    # CPU step
    opt_cpu.zero_grad()
    logits_cpu = model_cpu(x)
    loss_cpu = F.cross_entropy(logits_cpu, target)
    loss_cpu.backward()
    opt_cpu.step()

    # CUDA step (same data)
    opt_cuda.zero_grad()
    logits_cuda = model_cuda(x.cuda())
    loss_cuda = F.cross_entropy(logits_cuda, target.cuda())
    loss_cuda.backward()
    opt_cuda.step()

    if (step + 1) % 5 == 0:
        # Compare weights
        cpu_w = model_cpu.head.weight.data.flatten()[:3]
        cuda_w = model_cuda.head.weight.data.cpu().flatten()[:3]
        wdiff = (cpu_w - cuda_w).abs().max().item()
        print(f"  Step {step+1}: loss_cpu={loss_cpu.item():.4f}, loss_cuda={loss_cuda.item():.4f}, weight_diff={wdiff:.2e}")

# After 20 steps, compare full model outputs
x_test = torch.randn(8, 8, 16, 201)
out_cpu_final = model_cpu(x_test)
out_cuda_final = model_cuda(x_test.cuda()).cpu()
diff_final = (out_cpu_final - out_cuda_final).abs()
print(f"\nAfter 20 steps — output diff: max={diff_final.max().item():.4f}, mean={diff_final.mean().item():.4f}")
