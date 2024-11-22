# Deep Learning Principles - Slides
This repository contains the code and assets used to generate the slides and animations for the Deep Learning Principles talk. The talk originally took place in Seewiesen, 2024/11/21. The resulting presentaiton was designed to be understandable and appealing to a mixed audience of biologists.

A prerendered version of the presentation is now available at: https://ball-man.github.io/dl-principles-talk/.

Alternatively, keep reading to learn how to render and present these slides yourself.

## Installation
This project uses the manim (community edition) to render animation, and manim-slides to concatenate them into a viable presentation.

Install all the requirements:
```bash
pip install -r requirementx.txt
```

## Usage
Manim scenes have to be rendered beforehand, then manim slides can be used to visualize them. In practice:
```bash
manim [file.py] [SceneClass1] [SceneClass2] [...]
manim-slides [SceneClass1] [SceneClass2] [...]
```

Concretely for this project, use:
```bash
# Flags -ql set the rendering quality to "low", to save during debugging. Remove them for a 1080p render.
manim -ql slides.py AIFamily Logistic WhyLogistic LinearToNonLinear WhyNeuralNetworks Criterion GradientDescent BackProp ThankYou
```

And then:
```bash
manim-slides AIFamily Logistic WhyLogistic LinearToNonLinear WhyNeuralNetworks Criterion GradientDescent BackProp ThankYou
```
