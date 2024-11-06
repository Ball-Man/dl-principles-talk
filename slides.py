from itertools import chain

import numpy as np
from manim import *
from manim_slides.slide import Slide, ThreeDSlide

CLASS_A_COLOR = PURPLE
CLASS_B_COLOR = YELLOW
W1_COLOR = MAROON
W2_COLOR = PINK


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
        self.wait_time_between_slides = 0.1      # Fix incomplete animations

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
        self.add_fixed_orientation_mobjects(*dots_a, *dots_b)

        ## Slide: color dots
        self.play(dots_a_group.animate.set_color(CLASS_A_COLOR),
                  dots_b_group.animate.set_color(CLASS_B_COLOR))
        self.next_slide()

        ## Slide: create line
        line_w1 = ValueTracker(1)
        line_w2 = ValueTracker(1)
        line_b = ValueTracker(-9)

        line = Line()

        def line_updater(line: Line):
            # w1 * x + w2 * y + b = 0
            # x = (-b - w2 * y) / w1
            start_y = plot_range_y[0]
            start_x = (-line_b.get_value() - line_w2.get_value() * start_y) / line_w1.get_value()
            end_y = plot_range_y[1]
            end_x = (-line_b.get_value() - line_w2.get_value() * end_y) / line_w1.get_value()
            return line.become(Line(plane.c2p(start_x, start_y, 0), plane.c2p(end_x, end_y, 0)))

        line.add_updater(line_updater)

        # Create line equation
        equation_font_size = 30
        def get_line_equation(text: str):
            tex = MathTex(text, substrings_to_isolate=['w_1', 'x', 'w_2', 'y', 'b'])
            tex.set_color_by_tex('w_1', W1_COLOR)
            tex.set_color_by_tex('w_2', W2_COLOR)
            tex.font_size = equation_font_size
            return tex

        line_equation = get_line_equation('w_1x + w_2y + b = 0')

        def get_w1_tex() -> MathTex:
            label = MathTex(f'w_1 = {{{{ {line_w1.get_value():.2f} }}}}')
            label.submobjects[1].color = W1_COLOR
            label.font_size = equation_font_size
            return label

        def get_w2_tex() -> MathTex:
            label = MathTex(f'w_2 = {{{{ {line_w2.get_value():.2f} }}}}')

            label.submobjects[1].color = W2_COLOR
            label.font_size = equation_font_size
            return label

        w1_label = get_w1_tex().next_to(line_equation, DOWN)
        w2_label = get_w2_tex().next_to(w1_label, DOWN)
        equation_group = VGroup(line_equation, w1_label, w2_label).shift(UP * 3 + LEFT * 2)

        w_label_displacement_x = ValueTracker(DOWN[0])
        w_label_displacement_y = ValueTracker(DOWN[1])
        w_label_displacement_z = ValueTracker(DOWN[2])

        def w_label_displacement():
            return np.array(
                    [w_label_displacement_x.get_value(), w_label_displacement_y.get_value(),
                     w_label_displacement_z.get_value()])

        def tex_w1_updater(tex: MathTex):
            return (tex.become(get_w1_tex(), match_height=True)
                       .next_to(line_equation, w_label_displacement(), aligned_edge=LEFT))

        def tex_w2_updater(tex: MathTex):
            return (tex.become(get_w2_tex(), match_height=True)
                       .next_to(w1_label, w_label_displacement(), aligned_edge=LEFT))

        w1_label.add_updater(tex_w1_updater)
        w2_label.add_updater(tex_w2_updater)

        top_rect = SurroundingRectangle(equation_group, color=WHITE)
        top_rect.set_fill(config.background_color, 1.)

        self.play(Create(line), Create(top_rect), Write(line_equation), Write(w1_label),
                  Write(w2_label))
        self.next_slide(loop=True)

        ## Slide: plausible lines
        self.play(line_w1.animate.set_value(1.9), line_w2.animate.set_value(0.9),
                  line_b.animate.set_value(-22))
        self.play(line_w1.animate.set_value(1.2), line_w2.animate.set_value(1.5),
                  line_b.animate.set_value(-17))
        self.play(line_w1.animate.set_value(1), line_w2.animate.set_value(1),
                  line_b.animate.set_value(-9))
        self.next_slide()

        ## Slide: From decision boundary to plane
        # Disable temporarily the weights updaters to avoid animation artifacts
        w1_label.remove_updater(tex_w1_updater)
        w2_label.remove_updater(tex_w2_updater)

        old_line_equation = line_equation
        line_equation = get_line_equation('z = w_1x + w_2y + b').shift(UP * 3 + LEFT * 2)
        equation_group.remove(old_line_equation)
        equation_group.add(line_equation)
        self.play(TransformMatchingTex(old_line_equation, line_equation))

        w1_label.add_updater(tex_w1_updater)
        w2_label.add_updater(tex_w2_updater)

        self.next_slide()

        ## Slide: show dots on plane
        # Keep text oriented towards the camera
        self.add_fixed_orientation_mobjects(*equation_group.submobjects)

        def dot_z(dot: Dot):
            # z = w1 * x + w2 * y + b
            dot_x, dot_y = plane.p2c(dot.get_center())
            return (line_b.get_value() + line_w1.get_value() * dot_x
                    + line_w2.get_value() * dot_y) / 5

        # Before adding the updater, shift dots to the designated z
        # elegantly.
        self.move_camera(phi=PI / 2, added_anims=[
            *(dot.animate.set_z(dot_z(dot)) for dot in chain(dots_a, dots_b)),
            # Rotate accordingly the tex text
            line_equation.animate.shift(4 * OUT),
            w_label_displacement_x.animate.set_value(IN[0] * 2),
            w_label_displacement_y.animate.set_value(IN[1] * 2),
            w_label_displacement_z.animate.set_value(IN[2] * 2)])

        def dot_updater(dot: Dot):
            return dot.set_z(dot_z(dot))

        for dot in chain(dots_a, dots_b):
            dot.add_updater(dot_updater)
