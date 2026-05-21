"""
Project 29: Step 1 — Build a tiny vision encoder

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: (B, 3, 28, 28)
        x = self.proj(x)                  # (B, embed_dim, 7, 7)
        x = x.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)
        return x

class TinyViT(nn.Module):
    def __init__(self, depth=4, embed_dim=192, num_heads=6,
                 img_size=28, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size,
                                      in_chans=3, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
