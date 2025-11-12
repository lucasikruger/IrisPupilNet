# https://colab.research.google.com/drive/1IrOLXMQrkRI38afVcLJiGOCXKtd1c6po#scrollTo=wFlNFiWr9HSN

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if shapes mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, n_classes=3, base=32):
        super().__init__()
        self.inc = DoubleConv(3, base)             # -> base
        self.d1  = Down(base, base*2)              # -> base*2
        self.d2  = Down(base*2, base*4)            # -> base*4
        self.d3  = Down(base*4, base*8)            # -> base*8

        # after upsample: concat with skip => in_ch must be sum of both
        self.u1  = Up(base*8 + base*4, base*4)     # (x4 up base*8) + (skip x3 base*4)
        self.u2  = Up(base*4 + base*2, base*2)     # (x up base*4) + (skip x2 base*2)
        self.u3  = Up(base*2 + base,     base)     # (x up base*2) + (skip x1 base)

        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)    # base
        x2 = self.d1(x1)    # base*2
        x3 = self.d2(x2)    # base*4
        x4 = self.d3(x3)    # base*8

        x  = self.u1(x4, x3)  # -> base*4
        x  = self.u2(x,  x2)  # -> base*2
        x  = self.u3(x,  x1)  # -> base
        return self.out(x)


model = UNetSmall(n_classes=3, base=32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def iou_score(logits, y_true, num_classes=3):
    with torch.no_grad():
        preds = logits.argmax(1)
        ious = []
        for c in range(1, num_classes):  # ignore bg in mean IoU
            inter = ((preds==c) & (y_true==c)).sum().item()
            union = ((preds==c) | (y_true==c)).sum().item()
            ious.append(1.0 if union==0 else inter/union)
        return float(np.mean(ious))

best_val = 0.0
epochs = 20

for ep in range(1, epochs+1):
    model.train(); tr_loss = 0.0
    for x,y in tqdm(train_dl, desc=f"Train {ep}/{epochs}", leave=False):
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        tr_loss += loss.item()*x.size(0)
    tr_loss /= len(train_dl.dataset)

    model.eval(); val_loss = 0.0; val_iou = 0.0
    with torch.no_grad():
        for x,y in val_dl:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            val_loss += criterion(logits, y).item()*x.size(0)
            val_iou  += iou_score(logits, y)*x.size(0)
    val_loss /= len(val_dl.dataset)
    val_iou  /= len(val_dl.dataset)
    print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {val_loss:.4f} | IoU(iris+pupil) {val_iou:.3f}")
    if val_iou > best_val:
        best_val = val_iou
        torch.save(model.state_dict(), "/content/unet_mobius_best.pt")
        print("  ↳ saved /content/unet_mobius_best.pt")
