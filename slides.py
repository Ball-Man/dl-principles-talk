import numpy as np
from manim import *
from manim_slides.slide import Slide, ThreeDSlide

CLASS_A_COLOR = PURPLE
CLASS_B_COLOR = YELLOW


def static_slide(slide, tiny_time=0.1):
    """Apply a tiny wait animation to simulate a static slide."""
    slide.play(Wait(tiny_time))


class Welcome(Slide):

    def construct(self):
        text = Text('Deep Learning Principles')
        line = Line(2 * LEFT, 2 * RIGHT).next_to(text, DOWN)

        self.add(text)
        self.add(line)
        static_slide(self)

        self.next_slide()


def generate_points(rng: np.random.Generator, n, range_x, range_y):
    return np.concatenate((rng.uniform(*range_x, (n, 1)), rng.uniform(*range_y, (n, 1))), axis=1)


class Logistic(ThreeDSlide):
    rng = np.random.default_rng(213)

    def construct(self):
        ## Slide: build plane and dots
        plot_range_x = (-2, 15)
        plot_range_y = (-2, 10)
        plane = NumberPlane(plot_range_x, plot_range_y, x_length=15, y_length=8)
        plane.add_coordinates()

        dots_a_origins_coords = generate_points(self.rng, 8, (1, 6), (1, 4))
        dots_b_origins_coords = generate_points(self.rng, 8, (8, 14), (6, 9))

        dots_a_group = VGroup()
        dots_b_group = VGroup()
        dots_a = [Dot(plane.c2p(*coords2d)) for coords2d in dots_a_origins_coords]
        dots_b = [Dot(plane.c2p(*coords2d)) for coords2d in dots_b_origins_coords]

        dots_a_group.add(*dots_a)
        dots_b_group.add(*dots_b)

        self.play(Write(plane))
        self.play(GrowFromCenter(dots_a_group), GrowFromCenter(dots_b_group))

        self.next_slide()

        ## Slide: color dots
        self.play(dots_a_group.animate.set_color(CLASS_A_COLOR),
                  dots_b_group.animate.set_color(CLASS_B_COLOR))
        self.next_slide()
