"""
Project 5: Step 11 — Write the training loop yourself

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import logging

model = GPT(cfg, vocab_size).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

for step in range(cfg.max_steps):
    lr = get_lr(step, cfg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    xb, yb = get_batch("train", cfg)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    if step % cfg.eval_interval == 0:
        train_loss = estimate_loss("train")
        val_loss = estimate_loss("val")
        logging.info("step %d  train_loss %.4f  val_loss %.4f", step, train_loss, val_loss)
        sample_text(model)
        save_checkpoint(...)
