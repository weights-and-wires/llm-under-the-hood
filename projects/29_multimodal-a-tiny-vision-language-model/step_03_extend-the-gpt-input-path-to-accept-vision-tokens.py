"""
Project 29: Step 3 — Extend the GPT input path to accept vision tokens

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class GPT(nn.Module):
    def forward(self, idx=None, vision_embeds=None, targets=None):
        if vision_embeds is not None:
            # idx may still be present for the text portion
            text_embeds = self.tok_emb(idx) if idx is not None else None
            if text_embeds is not None:
                x = torch.cat([vision_embeds, text_embeds], dim=1)
            else:
                x = vision_embeds
        else:
            x = self.tok_emb(idx)

        T = x.shape[1]
        pos = self.pos_emb[:, :T, :]
        x = x + pos

        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        return logits, None

# For a batch with 49 image tokens and a caption of length T:
# targets shape: (B, 49 + T)
# First 49 positions: -100 (ignored)
# Remaining T positions: shifted caption tokens, normal cross-entropy
targets[:, :49] = -100
