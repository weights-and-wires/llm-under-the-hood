"""
Project 29: Step 6 — Run inference

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def caption(model, encoder, projection, image, tokenizer, max_new=20):
    model.eval()
    with torch.no_grad():
        # Encode the image
        vision_tokens = encoder(image.unsqueeze(0))
        vision_embeds = projection(vision_tokens)

        # Optional text prompt
        prompt_ids = tokenizer.encode("a photograph of")
        idx = torch.tensor([prompt_ids])

        # Generate
        for _ in range(max_new):
            logits, _ = model(idx=idx, vision_embeds=vision_embeds)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break

        return tokenizer.decode(idx[0].tolist())
