"""
Project 29: Step 4 — Pretrain on (image, caption) pairs

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

LABEL_TO_CAPTION = {
    0: "a photograph of an airplane.",
    1: "a photograph of an automobile.",
    2: "a photograph of a bird.",
    3: "a photograph of a cat.",
    4: "a photograph of a deer.",
    5: "a photograph of a dog.",
    6: "a photograph of a frog.",
    7: "a photograph of a horse.",
    8: "a photograph of a ship.",
    9: "a photograph of a truck.",
}

images, labels = next(loader)
captions = [LABEL_TO_CAPTION[y.item()] for y in labels]
input_ids, target_ids = encode_captions(captions, tokenizer)

vision_tokens = encoder(images)         # (B, 49, 192)
vision_embeds = projection(vision_tokens)  # (B, 49, 384)

# Pad targets with -100 in image positions
targets = pad_targets_with_image_mask(target_ids, num_image_tokens=49)

logits, loss = model(idx=input_ids,
                     vision_embeds=vision_embeds,
                     targets=targets)
loss.backward()
optimizer.step()
