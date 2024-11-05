from manim import *
from manim_slides.slide import Slide


class ExampleText(Slide):

    def construct(self):
        text = Tex('Hello everyone $2\pi$')
        self.play(Write(text))

        self.next_slide()
