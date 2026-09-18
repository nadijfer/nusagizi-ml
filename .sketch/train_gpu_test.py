import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def check_environment():
    print("=" * 60)
    print("🔍 SISTEM & INFORMASI GPU COLAB")
    print("=" * 60)
    print(f"PyTorch Version : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available  : {cuda_available}")

    if not cuda_available:
        print("⚠️  PERINGATAN: GPU tidak terdeteksi! Script akan berjalan di CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device Name     : {gpu_name}")
        print(f"Compute Cap     : {capability[0]}.{capability[1]}")
        print(f"Total VRAM      : {vram_gb:.2f} GB")

    print(f"Active Device   : {device}")
    print("=" * 60)
    return device


class SmallConvNet(nn.Module):
    """Model CNN kecil untuk klasifikasi gambar sintetis (3 channel, 32x32)"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_synthetic_dataloader(num_samples=10000, batch_size=64):
    """Membuat data sintetis agar tidak perlu download dataset eksternal"""
    print(f"\n📦 Menyiapkan dataset sintetis: {num_samples} gambar (3x32x32)...")
    # Gambar dummy acak berukuran (3, 32, 32)
    x = torch.randn(num_samples, 3, 32, 32)
    # Label acak 0-9
    y = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    return loader


def train(device, epochs=5, batch_size=64):
    loader = create_synthetic_dataloader(num_samples=12800, batch_size=batch_size)

    model = SmallConvNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n🚀 Memulai Proses Training...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_duration = time.time() - epoch_start_time
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        throughput = total / epoch_duration

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Waktu: {epoch_duration:.2f}s "
            f"| Kecepatan: {throughput:6.1f} samples/s "
            f"| Loss: {epoch_loss:.4f} "
            f"| Acc: {epoch_acc:.2f}%"
        )

    total_duration = time.time() - total_start_time
    print("-" * 60)
    print(f"⏱️  Total Durasi Training: {total_duration:.2f} detik")

    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"⚡ Peak GPU VRAM Terpakai : {peak_mem_mb:.2f} MB")

    # Tes Simpan Model
    save_path = "/tmp/small_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Checkpoint model berhasil disimpan ke: {save_path}")
    print("=" * 60)
    print("🎉 Tes GPU Colab Selesai dengan Sukses!")
    print("=" * 60)


if __name__ == "__main__":
    dev = check_environment()
    train(dev, epochs=5, batch_size=128)
