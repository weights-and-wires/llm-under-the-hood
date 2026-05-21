"""
Project 29: Step 2 — Build the projection layer

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

class Projection(nn.Module):
    def __init__(self, vision_dim=192, text_dim=384):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, text_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(text_dim, text_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
