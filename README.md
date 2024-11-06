## Installation
This project uses the manim (community edition) to render animation, and manim-slides to concatenate them into a viable presentation.

Install all the requirements:
```bash
pip install -r requirementx.txt
```

## Usage
Manim scenes have to be rendered before hand, then manim slides can be used to visualize them. In practice:
```bash
manim [file.py] [SceneClass1] [SceneClass2] [...]
manim-slides [SceneClass1] [SceneClass2] [...]
```

Concretely for this project, try:
```bash
# These are two of the scenes I'm working on at the moment
# Flags -ql set the rendering quality to "low", to save during debugging. Remove them for a 1080p render.
manim -ql slides.py Welcome Logistic
```

And then:
```bash
manim-slides Welcome Logistic
```
