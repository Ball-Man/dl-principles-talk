from itertools import chain, product
from collections.abc import Iterable, Callable
from functools import partial

import numpy as np
import scipy.interpolate
import scipy.optimize

from manim import *
from manim_slides.slide import Slide, ThreeDSlide

ML_COLOR = ORANGE
DL_COLOR = RED_C

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


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-value))


class AIFamily(Slide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations

        ## Slide: title
        title = Text('The AI Family')
        self.add(title)
        static_slide(self)
        self.next_slide()

        ## Slide: main AI set
        ai_set = RoundedRectangle(width=6, height=6).shift(DOWN)
        ai_set_label = Text('AI').move_to(ai_set, UL).shift(DR / 4)
        ai_set_label.font_size = 30
        self.play(LaggedStart(title.animate.to_edge(UP), AnimationGroup(Create(ai_set), Write(ai_set_label)), lag_ratio=0.5))
        self.next_slide()

        ## Slide: ML subset
        ml_subset = Circle(5 / 2, color=ML_COLOR).shift(DOWN)
        ml_subset_label = Text('ML').move_to(ml_subset, UL).shift(DR / 3 * 2)
        ml_subset_label.color = ml_subset.stroke_color
        ml_subset_label.font_size = 30
        self.play(Create(ml_subset), Write(ml_subset_label))
        self.next_slide()

        ## Slide: DL subset
        dl_subset = Circle(3 / 2, color=DL_COLOR).shift(DOWN)
        dl_subset_label = Text('DL').move_to(dl_subset, UL).shift(DR / 2)
        dl_subset_label.color = dl_subset.stroke_color
        dl_subset_label.font_size = 30
        self.play(Create(dl_subset), Write(dl_subset_label))
        self.next_slide()


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
        scene_rotation = ValueTracker(0)
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

        line_equation = get_line_equation('w_1x + w_2y + b = 0').shift(UP * 3 + LEFT * 2)

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

        w_label_displacement_vec = line_equation.get_critical_point(LEFT) + DOWN / 3
        w_label_displacement_x = ValueTracker(w_label_displacement_vec[0])
        w_label_displacement_y = ValueTracker(w_label_displacement_vec[1])
        w_label_displacement_z = ValueTracker(w_label_displacement_vec[2])
        w1_label = get_w1_tex().move_to(w_label_displacement_vec, LEFT)


        w2_label_displacement_vec = w1_label.get_critical_point(LEFT) + DOWN / 3
        w2_label_displacement_x = ValueTracker(w2_label_displacement_vec[0])
        w2_label_displacement_y = ValueTracker(w2_label_displacement_vec[1])
        w2_label_displacement_z = ValueTracker(w2_label_displacement_vec[2])
        w2_label = get_w2_tex().move_to(w2_label_displacement_vec, LEFT)
        equation_group = VGroup(line_equation, w1_label, w2_label)

        def w_label_displacement():
            return np.array(
                    [w_label_displacement_x.get_value(), w_label_displacement_y.get_value(),
                     w_label_displacement_z.get_value()])

        def w2_label_displacement():
            return np.array(
                    [w2_label_displacement_x.get_value(), w2_label_displacement_y.get_value(),
                     w2_label_displacement_z.get_value()])

        def tex_w1_updater(tex: MathTex):
            return (tex.become(get_w1_tex())
                       .rotate(scene_rotation.get_value(), RIGHT)
                       .move_to(w_label_displacement(), LEFT))

        def tex_w2_updater(tex: MathTex):
            return (tex.become(get_w2_tex())
                       .rotate(scene_rotation.get_value(), RIGHT)
                       .move_to(w2_label_displacement(), LEFT))

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
        def dot_z(dot: Dot):
            # z = w1 * x + w2 * y + b
            dot_x, dot_y = plane.p2c(dot.get_center())
            return (line_b.get_value() + line_w1.get_value() * dot_x
                    + line_w2.get_value() * dot_y) / 5

        line.remove_updater(line_updater)       # Prepare to remove the line

        # Must use an updater to animate the rotation, maybe due to some
        # bug.
        twod_line_equation = line_equation.copy().set_opacity(0)
        def line_equation_updater(eq: MathTex):
            return eq.become(twod_line_equation).set_opacity(1).rotate(scene_rotation.get_value(),
                                                                       RIGHT)
        line_equation.add_updater(line_equation_updater)

        # Before adding the updater, shift dots to the designated z
        # elegantly.
        w_label_displacement_vec += 3.5 * OUT
        self.move_camera(phi=PI / 2, added_anims=[
            *(dot.animate.set_z(dot_z(dot)) for dot in chain(dots_a, dots_b)),
            # Hide extra stuff from the scene
            FadeOut(line),
            FadeOut(top_rect),
            # Rotate accordingly the tex text
            twod_line_equation.animate.shift(4 * OUT),
            scene_rotation.animate.set_value(PI / 2),
            w_label_displacement_x.animate.set_value(w_label_displacement_vec[0]),
            w_label_displacement_y.animate.set_value(w_label_displacement_vec[1]),
            w_label_displacement_z.animate.set_value(w_label_displacement_vec[2]),
            w2_label_displacement_z.animate.set_value(3)])

        line_equation.remove_updater(line_equation_updater)

        def dot_updater(dot: Dot):
            return dot.set_z(dot_z(dot))

        for dot in chain(dots_a, dots_b):
            dot.add_updater(dot_updater)

        self.next_slide(loop=True)

        ## Slide: move the decision boundary, in z view this time
        self.play(line_w1.animate.set_value(1.9), line_w2.animate.set_value(0.9),
                  line_b.animate.set_value(-22))
        self.play(line_w1.animate.set_value(1.2), line_w2.animate.set_value(1.5),
                  line_b.animate.set_value(-17))
        self.play(line_w1.animate.set_value(1), line_w2.animate.set_value(1),
                  line_b.animate.set_value(-9))
        self.next_slide()

        ## Slide: sigmoid
        old_line_equation = line_equation
        line_equation = (get_line_equation(r'z = \sigma(w_1x + w_2y + b)')
                         .move_to(line_equation, LEFT).rotate(PI / 2, axis=RIGHT))

        sigmoid_label = (MathTex(r'\sigma(t) = \frac{1}{1 + e^{-t}}')
                         .next_to(line_equation, 3 * RIGHT)
                         .shift(2 * UP)
                         .rotate(PI / 2, RIGHT))
        sigmoid_label.font_size = equation_font_size

        # Temp remove updater to apply sigmoid to all dots
        for dot in chain(dots_a, dots_b):
            dot.remove_updater(dot_updater)

        dots_sig_z = sigmoid(np.array([dot.get_z() for dot in chain(dots_a, dots_b)]))

        self.play(TransformMatchingTex(old_line_equation, line_equation),
                  Write(sigmoid_label),
                  # Sigmoid the dots
                  *(dot.animate.set_z(new_z) for dot, new_z in zip(chain(dots_a, dots_b),
                                                                   dots_sig_z)))
        self.next_slide(loop=True)

        def dot_sigmoid_updater(dot: Dot):
            return dot.set_z(sigmoid(dot_z(dot)))

        for dot in chain(dots_a, dots_b):
            dot.add_updater(dot_sigmoid_updater)

        self.play(line_w1.animate.set_value(1.9), line_w2.animate.set_value(0.9),
                  line_b.animate.set_value(-22))
        self.play(line_w1.animate.set_value(1.2), line_w2.animate.set_value(1.5),
                  line_b.animate.set_value(-17))
        self.play(line_w1.animate.set_value(1), line_w2.animate.set_value(1),
                  line_b.animate.set_value(-9))
        self.next_slide()


class WhyLogistic(Slide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations
        body_text_size = 30

        ## Slide: title
        why_title = Text('Why Logistic Regression?')
        static_slide(self)
        self.add(why_title)
        self.next_slide()

        ## Slide: some positive aspects
        # Prepare body
        intuitive_text = Paragraph('Linear relationships are intuitive', font_size=body_text_size)
        lightweight_text = Text('Lightweight computationally', font_size=body_text_size)
        prob_output_text = Text('Probabilistic output', font_size=body_text_size)
        logistic_regression_body = VGroup(intuitive_text, lightweight_text, prob_output_text)
        logistic_regression_body.arrange(3 * DOWN, aligned_edge=LEFT).shift(3 * LEFT)

        self.play(AnimationGroup(why_title.animate.to_edge(UP),
                                 Write(logistic_regression_body), lag_ratio=0.6))
        self.next_slide()


def neural_net_connection_arrow(*args, **kwargs) -> Arrow:
    """Build an arrow suitable for neural net links."""
    return Arrow(*args, stroke_width=5,  # Fixed line width
            tip_length=0.15,  # Fixed tip length
            max_tip_length_to_length_ratio=1, **kwargs)


def all_arrows(from_objects: Iterable[VMobject], to_objects: Iterable[VMobject],
               line_factory=neural_net_connection_arrow) -> list[VMobject]:
    """Get lines connecting the objects in from_ to the objects in to_.

    This creates all connections.
    """
    lines = []
    for from_, to in product(from_objects, to_objects):
        lines.append(line_factory(from_.get_critical_point(RIGHT), to.get_critical_point(LEFT)))
    return lines


def pair_arrows(from_objects: Iterable[VMobject], to_objects: Iterable[VMobject],
                line_factory=neural_net_connection_arrow) -> list[VMobject]:
    """Get lines connecting the objects in from_ to the objects in to_.

    This creates just the pairwise connections. So
    `len(from_objects) == len(to_objects)` is assumed.
    """
    lines = []
    for from_, to in zip(from_objects, to_objects):
        lines.append(line_factory(from_.get_critical_point(RIGHT), to.get_critical_point(LEFT)))
    return lines


class ColorGenerator:
    """Generate pseudorandom HSV colors."""

    def __init__(self, seed, saturation=1, value=0.5):
        self.rng = np.random.default_rng(seed)
        self.saturation = saturation
        self.value = value

    def __call__(self) -> ManimColor:
        return ManimColor.from_hsv((self.rng.random(), self.saturation, self.value))


class LinearToNonLinear(Slide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations
        formula_font_size = 40
        perceptron_color = RED
        color_generator = ColorGenerator(42)
        layer_1_colors = [color_generator() for _ in range(3 * 2)]
        layer_2_colors = [color_generator() for _ in range(2 * 2)]

        ## Slide: title
        title = Text('Logistic Regressor')
        self.add(title)

        static_slide(self)
        self.next_slide()

        ## Slide: Function definition
        function_def = MathTex(r'f: \mathbb{R}^2 \to \mathbb{R}').shift(UP)
        function_def.font_size = formula_font_size

        regressor_formula = MathTex(r'f(x, y) = \sigma(w_xx + w_yy + b) = output',
                                    substrings_to_isolate=[r'f(x, y) = \sigma(w_xx + w_yy', ')'])
        regressor_formula.font_size = formula_font_size

        self.play(title.animate.to_edge(UP), Write(function_def), Write(regressor_formula))

        self.next_slide()

        ## Slide: get rid of b
        regressor_formula_no_b = MathTex(
            r'f(x, y) = \sigma(w_xx + w_yy) = output',
            substrings_to_isolate=[r'f(x, y) = \sigma(w_xx + w_yy', ')'])
        regressor_formula_no_b.font_size = formula_font_size

        self.play(TransformMatchingTex(regressor_formula, regressor_formula_no_b))
        regressor_formula = regressor_formula_no_b

        self.next_slide()

        ## Slide: matrix form
        # Parameters used later on for consistency
        perceptron_group_shift = DOWN * 2
        matrix_form_shift = UP

        regressor_matrix_formula = MathTex(
            r'''
            f \left( \left[ \begin{array}{c} x \\ y \end{array} \right] \right) =
            \sigma\left(
            \left[ \begin{array}{cc}
            w_x & w_y
            \end{array} \right]
            \left[ \begin{array}{c}
            x \\
            y
            \end{array} \right]
            \right)
            = output
            '''
        )
        regressor_matrix_formula.font_size = formula_font_size

        # Animate the expression
        self.play(TransformMatchingShapes(regressor_formula, regressor_matrix_formula))
        self.next_slide()

        ## Slide: the perceptron
        perceptron = VGroup(Circle(0.2, color=perceptron_color), Text('+', color=perceptron_color))
        x_label = MathTex('x').next_to(perceptron, LEFT).shift(LEFT + 0.5 * UP)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT + 0.5 * DOWN)
        output_label = MathTex('output')
        output_label.font_size = formula_font_size
        output_label.next_to(perceptron, RIGHT).shift(RIGHT)
        x_y_lines = all_arrows((x_label, y_label), (perceptron,))
        output_line = neural_net_connection_arrow(perceptron.get_critical_point(RIGHT),
                                                  output_label.get_critical_point(LEFT))
        perceptron_group = VGroup(perceptron, x_label, y_label, *x_y_lines, output_label,
                                  output_line).to_edge(LEFT).shift(perceptron_group_shift)

        self.play(function_def.animate.to_edge(LEFT),
                  regressor_matrix_formula.animate.to_edge(LEFT))
        self.play(function_def.animate.shift(2 * UP),
                  regressor_matrix_formula.animate.shift(matrix_form_shift),
                  Write(perceptron_group, lag_ratio=0))
        self.add(index_labels(regressor_matrix_formula[0]).set_color(RED))
        self.next_slide()

        ## Slide: associate weights with connections in visual model
        w_x_group = VGroup(*regressor_matrix_formula[0][11:13])
        w_y_group = VGroup(*regressor_matrix_formula[0][13:15])

        floating_w_x = w_x_group.copy()
        floating_w_y = w_y_group.copy()

        self.play(floating_w_x.animate.next_to(x_y_lines[0], UP),
                  floating_w_y.animate.next_to(x_y_lines[1], DOWN))
        self.play(w_x_group.animate.set_color(layer_1_colors[0]),
                  floating_w_x.animate.set_color(layer_1_colors[0]),
                  w_y_group.animate.set_color(layer_1_colors[1]),
                  floating_w_y.animate.set_color(layer_1_colors[1]),
                  x_y_lines[0].animate.set_color(layer_1_colors[0]),
                  x_y_lines[1].animate.set_color(layer_1_colors[1]))
        self.play(floating_w_x.animate.become(x_y_lines[0]),
                  floating_w_y.animate.become(x_y_lines[1]))
        # Silently remove the overlapping arrows
        self.play(FadeOut(floating_w_x), FadeOut(floating_w_y))
        self.next_slide()

        ## Slide: multivaried function
        # Update the matrix form
        regressor_matrix_formula_multi = MathTex(
            r'''
            f \left( \left[ \begin{array}{c} x \\ y \\ z \end{array} \right] \right) =
            \sigma\left(
            \left[ \begin{array}{ccc}
            w_x & w_y & w_z
            \end{array} \right]
            \left[ \begin{array}{c}
            x \\
            y \\
            z
            \end{array} \right]
            \right)
            = output
            '''
        )
        regressor_matrix_formula_multi.font_size = formula_font_size
        regressor_matrix_formula_multi.to_edge(LEFT).shift(matrix_form_shift)

        # Update function def
        function_def_multi = MathTex(r'f: \mathbb{R}^3 \to \mathbb{R}')
        function_def_multi.font_size = formula_font_size
        function_def_multi.to_edge(LEFT).shift(3 * UP)

        # Update perceptron
        perceptron = VGroup(Circle(0.2, color=perceptron_color), Text('+', color=perceptron_color))
        x_label = MathTex('x').next_to(perceptron, LEFT).shift(UL)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT)
        z_label = MathTex('z').next_to(perceptron, LEFT).shift(DL)
        output_label = MathTex('output')
        output_label.font_size = formula_font_size
        output_label.next_to(perceptron, RIGHT).shift(RIGHT)
        x_y_lines = all_arrows((x_label, y_label, z_label), (perceptron,))
        output_line = Arrow(perceptron.get_critical_point(RIGHT),
                            output_label.get_critical_point(LEFT))
        perceptron_group_multi = VGroup(perceptron, x_label, y_label, z_label, *x_y_lines,
                                        output_label, output_line).to_edge(LEFT).shift(perceptron_group_shift)

        self.play(TransformMatchingShapes(regressor_matrix_formula,
                                          regressor_matrix_formula_multi),
                  TransformMatchingShapes(function_def, function_def_multi),
                  FadeTransform(perceptron_group, perceptron_group_multi))
        regressor_matrix_formula = regressor_matrix_formula_multi
        function_def = function_def_multi
        perceptron_group = perceptron_group_multi
        self.next_slide()

        ## Slide: more perceptrons (vector valued)
        # Update the matrix form
        regressor_matrix_formula_multi = MathTex(
            r'''
            f \left( \left[ \begin{array}{c} x \\ y \\ z \end{array} \right] \right) =
            \sigma\left(
            \left[ \begin{array}{ccc}
            w_{x1} & w_{y1} & w_{z1} \\
            w_{x2} & w_{y2} & w_{z2}
            \end{array} \right]
            \left[ \begin{array}{c}
            x \\
            y \\
            z
            \end{array} \right]
            \right)
            =
            \left[ \begin{array}{c}
            output_1 \\
            output_2 \\
            \end{array} \right]
            '''
        )
        regressor_matrix_formula_multi.font_size = formula_font_size
        regressor_matrix_formula_multi.to_edge(LEFT).shift(matrix_form_shift)

        # Update function def
        function_def_multi = MathTex(r'f: \mathbb{R}^3 \to \mathbb{R}^2')
        function_def_multi.font_size = formula_font_size
        function_def_multi.to_edge(LEFT).shift(3 * UP)

        # Update perceptron
        perceptron = VGroup(Circle(0.2, color=perceptron_color),
                            Text('+', color=perceptron_color)).shift(DOWN * 0.5)
        perceptron_2 = VGroup(Circle(0.2, color=perceptron_color),
                              Text('+', color=perceptron_color)).shift(UP * 0.5)
        # perceptron_3 = Circle(0.2).shift(UP)
        x_label = MathTex('x').next_to(perceptron, LEFT).shift(UL + UP * 0.5)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT + UP * 0.5)
        z_label = MathTex('z').next_to(perceptron, LEFT).shift(DL + UP * 0.5)
        output_label = MathTex('output_1')
        output_label_2 = MathTex('output_2')
        # output_label_3 = MathTex(
        #     r'f \left( \left[ \begin{array}{c} \vdots \end{array} \right] \right)_3'
        # ).shift(DOWN)
        for label in (output_label, output_label_2):        # , output_label_3
            label.font_size = formula_font_size
            label.next_to(perceptron, RIGHT).shift(RIGHT)
        output_label.shift(UP)
        # output_label_3.shift(DOWN)

        x_y_lines = all_arrows((x_label, y_label, z_label), (perceptron, perceptron_2))
        #                                                     perceptron_3))
        output_lines = pair_arrows((perceptron, perceptron_2),          # perceptron_3
                                   (output_label_2, output_label))      # output_label_3
        perceptron_group_multi = VGroup(perceptron, perceptron_2,       # perceptron_3,
                                        x_label, y_label, z_label,
                                        *x_y_lines, output_label, output_label_2, # output_label_3,
                                        *output_lines).to_edge(LEFT).shift(perceptron_group_shift)

        self.play(TransformMatchingShapes(regressor_matrix_formula,
                                          regressor_matrix_formula_multi),
                  TransformMatchingShapes(function_def, function_def_multi),
                  FadeTransform(perceptron_group, perceptron_group_multi))
        regressor_matrix_formula = regressor_matrix_formula_multi
        function_def = function_def_multi
        perceptron_group = perceptron_group_multi
        self.next_slide()

        ## Slide: more layers
        # Update the matrix form
        regressor_matrix_formula_multi = MathTex(
            r'''
            f \left( \left[ \begin{array}{c} x \\ y \\ z \end{array} \right] \right) =
            \sigma\left(
            \left[ \begin{array}{cc}
            w_{B1} & w_{B2} \\
            w_{B3} & w_{B4}
            \end{array} \right]
            \sigma\left(
            \left[ \begin{array}{ccc}
            w_{x1} & w_{y1} & w_{z1} \\
            w_{x2} & w_{y2} & w_{z2}
            \end{array} \right]
            \left[ \begin{array}{c}
            x \\
            y \\
            z
            \end{array} \right]
            \right) \right)
            =
            \left[ \begin{array}{c}
            output_1 \\
            output_2 \\
            \end{array} \right]
            '''
        )
        regressor_matrix_formula_multi.font_size = 35       # formula_font_size
        regressor_matrix_formula_multi.to_edge(LEFT).shift(matrix_form_shift)

        # Update function def
        function_def_multi = MathTex(r'f: \mathbb{R}^3 \to \mathbb{R}^2')
        function_def_multi.font_size = formula_font_size
        function_def_multi.to_edge(LEFT).shift(3 * UP)

        # Update perceptron
        perceptron = VGroup(Circle(0.2, color=perceptron_color),
                            Text('+', color=perceptron_color)).shift(DOWN * 0.5)
        perceptron_2 = VGroup(Circle(0.2, color=perceptron_color),
                              Text('+', color=perceptron_color)).shift(UP * 0.5)
        perceptron_2_1 = VGroup(Circle(0.2, color=perceptron_color),
                                Text('+', color=perceptron_color)).next_to(perceptron, RIGHT).shift(RIGHT)
        perceptron_2_2 = VGroup(Circle(0.2, color=perceptron_color),
                                Text('+', color=perceptron_color)).next_to(perceptron_2, RIGHT).shift(RIGHT)

        x_label = MathTex('x').next_to(perceptron, LEFT).shift(UL + UP * 0.5)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT + UP * 0.5)
        z_label = MathTex('z').next_to(perceptron, LEFT).shift(DL + UP * 0.5)
        output_label = MathTex('output_1')
        output_label_2 = MathTex('output_2')
        # output_label_3 = MathTex(
        #     r'f \left( \left[ \begin{array}{c} \vdots \end{array} \right] \right)_3'
        # ).shift(DOWN)
        for label in (output_label, output_label_2):        # output_label_3
            label.font_size = formula_font_size
            label.next_to(perceptron_2_1, RIGHT).shift(RIGHT)
        output_label.shift(UP)
        # output_label_3.shift(DOWN)

        x_y_lines = (
            all_arrows((x_label, y_label, z_label), (perceptron, perceptron_2))     # perceptron_3
            + all_arrows((perceptron, perceptron_2),            # perceptron_3
                         (perceptron_2_2, perceptron_2_1))      # perceptron_2_3
        )
        output_lines = pair_arrows((perceptron_2_1, perceptron_2_2),        # perceptron_2_3
                                   (output_label_2, output_label))          # output_label_3
        perceptron_group_multi = VGroup(perceptron, perceptron_2,           # perceptron_3
                                        perceptron_2_1, perceptron_2_2,     # perceptron_2_3,
                                        x_label, y_label, z_label,
                                        *x_y_lines, output_label, output_label_2, # output_label_3,
                                        *output_lines).to_edge(LEFT).shift(perceptron_group_shift)


        new_title = Text('Neural Network').to_edge(UP)
        self.play(AnimationGroup(Unwrite(title), Write(new_title, reverse=True), lag_ratio=0.6),
                  TransformMatchingShapes(regressor_matrix_formula,
                                          regressor_matrix_formula_multi),
                  TransformMatchingShapes(function_def, function_def_multi),
                  FadeTransform(perceptron_group, perceptron_group_multi))
        regressor_matrix_formula = regressor_matrix_formula_multi
        function_def = function_def_multi
        perceptron_group = perceptron_group_multi
        # Unwrite with(out) reverse is apparently bugged. So I write
        # reverse instead (it's almost more classy this way anyway), but
        # writing in reverse is also buggy, causing the text to
        # disappear immediately after the animation. Add the title here
        # to prevent this from happening.
        self.add(new_title)
        title = new_title
        self.next_slide()

        ## Slide: let's simplify
        regressor_concise_formula = MathTex(r'f(v) = \sigma(\,B\: \sigma(\,A v\,)\,)')
        regressor_concise_formula.font_size = formula_font_size
        regressor_concise_formula.to_edge(LEFT).shift(matrix_form_shift)

        self.play(TransformMatchingShapes(regressor_matrix_formula_multi,
                                          regressor_concise_formula))

        self.next_slide()

        ## Slide: going deep
        new_title = Text('Deep Neural Network').to_edge(UP)
        regressor_concise_formula_deep1 = MathTex(
            r'f(v) = \sigma(\,C\: \sigma(\,B\: \sigma(\,A v\,)\,)\,)'
        ).to_edge(LEFT).shift(matrix_form_shift)
        regressor_concise_formula_deep2 = MathTex(
            r'f(v) = \sigma(\,D\:\sigma(\,C\: \sigma(\,B\: \sigma(\,A v\,)\,)\,)\,)'
        ).to_edge(LEFT).shift(matrix_form_shift)
        regressor_concise_formula_deep3 = MathTex(
            r'''
            f(v) = \sigma(\,E\: \sigma(\,D\: \sigma(\,C\: \sigma(\,B\: \sigma(\,A v\,)
            \,)\,)\,)\,)
            '''
        ).to_edge(LEFT).shift(matrix_form_shift)

        self.play(FadeTransform(title, new_title),
                  TransformMatchingShapes(regressor_concise_formula,
                                          regressor_concise_formula_deep1),
                  FadeOut(perceptron_group))
        self.play(AnimationGroup(Wait(0.5),
                                 TransformMatchingShapes(regressor_concise_formula_deep1,
                                                         regressor_concise_formula_deep2),
                                 lag_ratio=1))
        self.play(AnimationGroup(Wait(0.5),
                                 TransformMatchingShapes(regressor_concise_formula_deep2,
                                                         regressor_concise_formula_deep3),
                                 lag_ratio=1))

        # TODO: if we have time, show also the network expansion

        self.next_slide()


class WhyNeuralNetworks(Slide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations
        body_text_size = 30

        ## Slide: title
        why_title = Text('Why Neural Networks?')
        static_slide(self)
        self.add(why_title)
        self.next_slide()

        ## Slide: universal approximators
        universal_approximators_text = Text(
            'Multilayer feedforward networks are universal\napproximators (Hornik & Stinchcombe & White, 1989)',
            font_size=body_text_size
        )
        flexibility_text = Text('Computational cost trade off', font_size=body_text_size)
        prob_output_text = Text('Probabilistic Output', font_size=body_text_size)

        neural_network_body = VGroup(universal_approximators_text, flexibility_text, prob_output_text)
        neural_network_body.arrange(3 * DOWN, aligned_edge=LEFT).to_edge(LEFT)

        self.play(AnimationGroup(why_title.animate.to_edge(UP),
                                 Write(neural_network_body), lag_ratio=0.6))
        self.next_slide()


def grid_position(x_index, y_index, horizontal_spacing=RIGHT,
                  vertical_spacing=DOWN, origin=ORIGIN) -> np.ndarray:
    """Get grid position given table space coordinates."""
    return origin + x_index * horizontal_spacing + y_index * vertical_spacing


def place_in_grid(obj: Mobject, x_index, y_index, *args, aligned_edge=RIGHT, **kwargs) -> Mobject:
    """Move an object in a grid position given table space coordinates."""
    return obj.move_to(grid_position(x_index, y_index, *args, **kwargs), aligned_edge=aligned_edge)


class Criterion(Slide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations

        ## Slide: title
        title = Text('How to Train Your Network')
        self.add(title)
        static_slide(self)
        self.next_slide()

        ## Slide: a network
        function_def = MathTex(r'f: \mathbb{R} \to \mathbb{R}').shift(UP)
        regressor_formula = MathTex(r'f(x) = \sigma(\,W_2\: \sigma(\,W_1 x\,)\,)',
                                    substrings_to_isolate=['f(x)'])

        self.play(FadeOut(title))
        self.play(Write(function_def), Write(regressor_formula))
        self.next_slide()

        ## Slide: the data
        # Define data
        data = np.array(((-2, 1, 14, -7, 5), (1, 0, 0, 1, 0)))
        label_map = ['A', 'B']

        # Define grid
        local_grid_config = {'origin': UP + 5 * LEFT, 'horizontal_spacing': 1.5 * RIGHT}
        local_grid = partial(place_in_grid, **local_grid_config)
        local_grid_position = partial(grid_position, **local_grid_config)

        table_group = VGroup()

        # Prepare header
        header_line = Line(local_grid_position(-1, -0.5), local_grid_position(5, -0.5))
        lines_group = VGroup(header_line)

        table_group.add(
            # Add header labels
            local_grid(MathTex('x'), 0, -1),
            local_grid(MathTex('y'), 1, -1))

        labels_columns_group = VGroup()         # Used later to morph these

        # Populate table with initial data
        for y_index, (x, y) in enumerate(data.T):
            table_group.add(local_grid(MathTex(str(x)), 0, y_index))
            y_label = local_grid(MathTex(label_map[y]), 1, y_index, aligned_edge=ORIGIN)
            labels_columns_group.add(y_label)
            table_group.add(y_label)

        for x_index, _ in enumerate(data):
            column_line = Line(local_grid_position(x_index + 0.5, -1.5),
                               local_grid_position(x_index + 0.5, data.shape[1]))
            lines_group.add(column_line)

        self.play(AnimationGroup(function_def.animate.to_corner(UL),
                                 regressor_formula.animate.to_edge(UP),
                                 lag_ratio=0.1))
        self.play(Create(lines_group, lag_ratio=0.2))
        self.play(Write(table_group))
        self.next_slide()

        ## Slide: from class label to 0-1
        numeric_labels_columns = VGroup(
            *(local_grid(MathTex(str(y)), 1, y_index, aligned_edge=ORIGIN)
              for y_index, y in enumerate(data[1]))
        )

        self.play(FadeTransform(labels_columns_group, numeric_labels_columns))
        table_group.remove(*labels_columns_group.submobjects)
        table_group.add(*numeric_labels_columns.submobjects)
        self.next_slide()

        ## Slide: predictions
        predictions = np.array([0.92, 0.4, 0.12, 0.7, 0.9])

        prediction_line = Line(local_grid_position(2.5, -1.5),
                               local_grid_position(2.5, data.shape[1]))
        lines_group.add(prediction_line)

        fx_header = local_grid(MathTex('f(x)'), 2, -1, aligned_edge=ORIGIN)
        predictions_column_group = VGroup()
        for y_index, prediction in enumerate(predictions):
            predictions_column_group.add(local_grid(MathTex(str(prediction)), 2, y_index,
                                                    aligned_edge=ORIGIN))

        self.play(Create(prediction_line),
                  TransformMatchingShapes(regressor_formula.submobjects[0].copy(), fx_header))
        self.play(Write(predictions_column_group))
        self.next_slide()

        ## Slide: criterion
        criterion_results = np.abs(data[1] - predictions)
        print(criterion_results)

        criterion = local_grid(MathTex('l_f(x, y) = |f(x) - y|'), 2.75, -1, aligned_edge=LEFT)
        criterion_results_group = VGroup()
        for y_index, criterion_value in enumerate(criterion_results):
            criterion_results_group.add(local_grid(MathTex(f'{criterion_value:.2f}'), 3, y_index,
                                                   aligned_edge=ORIGIN))

        reduced_criterion = local_grid(
            MathTex(f'{{{{ \sum_{{x, y}} l_f(x, y) }}}} = {criterion_results.sum():.2f}'),
            4, 2, aligned_edge=LEFT)
        reduction_arrow_group = VGroup(*all_arrows(criterion_results_group.submobjects,
                                                   (reduced_criterion,)))

        self.play(Write(criterion), Write(criterion_results_group))
        self.play(Create(reduction_arrow_group), Write(reduced_criterion))
        self.next_slide()

        ## Slide: objective
        objective = MathTex(r'\min_{W_1, W_2} {{ \sum_{x, y} l_f(x, y) }}').shift(DOWN)
        objective_highlight = SurroundingRectangle(objective)

        self.play(FadeOut(lines_group), FadeOut(table_group), FadeOut(predictions_column_group),
                  FadeOut(reduction_arrow_group), FadeOut(criterion_results_group),
                  FadeOut(fx_header))
        self.play(TransformMatchingTex(reduced_criterion, objective), criterion.animate.move_to(UP))
        self.play(Create(objective_highlight))
        self.next_slide()


class GradientDescent(ThreeDSlide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations

        local_minima_color = RED
        slope_color = YELLOW
        body_font_size = 30

        title = Text('Finding minima:\nGradient Descent')
        self.add(title)
        static_slide(self)
        self.next_slide()

        # x, y pairs
        f_pairwise = ((-2, 2), (1, 4), (5, 3), (6, 1), (9, 5), (10, 4), (12, 5),
                      (15, 3,), (15.5, 3.5), (16, 2), (17, 0.5),
                      (19, 4), (21, 5), (22, 10), (27, 0), (27.5, 1), (28, -0.5), (30, 3),
                      (33, -2),
                      (1_000_000, -100))
        x_points, f_values = zip(*f_pairwise)

        # Create a cubic spline interpolator
        f = scipy.interpolate.InterpolatedUnivariateSpline(x_points, f_values)
        f_d = f.derivative()
        local_minimum_1 = scipy.optimize.minimize_scalar(f, bounds=(6, 6.5), method='bounded').x
        local_minimum_2 = scipy.optimize.minimize_scalar(f, bounds=(16, 17), method='bounded').x
        local_minimum_shallow = scipy.optimize.minimize_scalar(f, bounds=(9, 11),
                                                               method='bounded').x


        plot_range_x = (-1.5, 50)
        plot_range_y = (-1.5, 10)
        plane = NumberPlane(plot_range_x, plot_range_y, x_length=50, y_length=10)
        plane.to_corner(DL, 0)
        plane.add_coordinates()
        function_plot = plane.plot(f)

        self.play(FadeOut(title))
        self.play(Write(plane))
        self.play(Write(function_plot))

        self.next_slide()

        local_argmin = plane.c2p(local_minimum_1, 0, 0)
        on_plot_local_argmin = plane.i2gp(local_minimum_1, function_plot)
        local_minima_dot = Dot(local_argmin, color=local_minima_color)

        self.play(Create(local_minima_dot))
        # self.play(local_minima_dot.animate.move_to(on_plot_local_argmin))
        self.play(Transform(local_minima_dot,
                            Line(on_plot_local_argmin + LEFT * 10,
                                 on_plot_local_argmin + RIGHT * 10,
                                 color=local_minima_color)))

        self.next_slide()

        ## Slide: slide and show a minimum
        camera_shift = 10 * RIGHT
        self.move_camera(frame_center=camera_shift,
                         added_anims=[local_minima_dot.animate.shift(camera_shift)])
        self.next_slide()

        ## Slide: move bar to the new minimum
        on_plot_local_argmin_2 = plane.i2gp(local_minimum_2, function_plot)
        self.play(local_minima_dot.animate.move_to(on_plot_local_argmin_2))
        self.next_slide()

        ## Slide: move camera again and show that it is really bad
        camera_shift *= 2.5
        self.move_camera(zoom=0.8, frame_center=camera_shift, run_time=6,
                         added_anims=[local_minima_dot.animate.shift(15 * RIGHT)])
        self.move_camera(frame_center=ORIGIN, zoom=1, added_anims=[FadeOut(local_minima_dot)])
        self.next_slide()

        ## Slide: cover a bit with black to write things, and write
        cover_rectangle = Rectangle()
        cover_rectangle.stretch_to_fit_width(16)
        cover_rectangle.to_corner(UL, 0.)
        cover_rectangle.set_stroke(opacity=0.)
        cover_rectangle.set_fill(config.background_color, 1.)

        minima_hard_text = Tex('Global minima are hard, let\'s keep it local', font_size=30)
        minima_hard_text.to_corner(UL)

        self.play(Create(cover_rectangle))
        self.play(Write(minima_hard_text))

        self.next_slide()

        ## Slide: Choose a point
        pointed_x = ValueTracker(8)
        current_location = plane.i2gp(pointed_x.get_value(), function_plot)
        current_dot = Dot(color=local_minima_color).move_to(current_location)

        self.play(FadeIn(current_dot, target_position=current_location + 3 * UP))
        self.next_slide()

        ## Slides: show dot and slope
        current_dot.add_updater(lambda dot: dot.move_to(plane.i2gp(pointed_x.get_value(),
                                                                   function_plot)))
        example_slope_x1 = 7.3
        example_slope_x2 = 8.7
        example_slope_on_curve_1 = plane.i2gp(example_slope_x1, function_plot)
        example_slope_on_curve_2 = plane.i2gp(example_slope_x2, function_plot)

        slope_from_1 = example_slope_on_curve_1 - UP * 2
        slope_from_2 = example_slope_on_curve_2 - UP * 4
        slope_sel_1 = DashedLine(slope_from_1, example_slope_on_curve_1, color=slope_color)
        slope_sel_2 = DashedLine(slope_from_2, example_slope_on_curve_2, color=slope_color)
        slope_line = Line(example_slope_on_curve_1, example_slope_on_curve_2, stroke_width=6,
                          color=slope_color)

        example_slope_dir_vec = example_slope_on_curve_1 - example_slope_on_curve_2
        example_slope_dir = np.arctan2(example_slope_dir_vec[1], example_slope_dir_vec[0])

        slope_tip = (ArrowTriangleTip(fill_opacity=1, stroke_width=0, width=0.2, color=slope_color)
                     .move_to(example_slope_on_curve_1)
                     .rotate(PI + example_slope_dir))
        # Fix tip to the line
        slope_tip.add_updater(lambda tip: tip.move_to(slope_line.get_start()))

        # Update the line so that it always reflects the real slope from now on
        def slope_updater(line: Line):
            angle = np.arctan(f_d(pointed_x.get_value()))
            # For convenience, build the line at origin with unitary
            # size, then move it.
            return line.become(Line(ORIGIN, ORIGIN + np.array([np.cos(angle), np.sin(angle), 0]),
                                    stroke_width=6, color=slope_color)
                                    .move_to(plane.i2gp(pointed_x.get_value(), function_plot)))

        self.move_camera(frame_center=current_dot , zoom=2)
        self.next_slide()

        self.play(Create(slope_sel_1))
        self.play(Create(slope_sel_2))
        self.next_slide()

        self.play(Create(slope_line))
        self.next_slide()

        self.play(Create(slope_tip))
        self.next_slide()

        ## Slide: back to original view
        self.move_camera(frame_center=ORIGIN , zoom=1,
                         added_anims=[slope_updater(slope_line.animate),
                                      FadeOut(slope_tip),
                                      FadeOut(slope_sel_1), FadeOut(slope_sel_2)])
        self.next_slide()

        ## Slide: descend
        slope_line.add_updater(slope_updater)

        self.play(pointed_x.animate.set_value(local_minimum_1))
        self.next_slide()

        ## Slide: new point, from left
        slope_line.remove_updater(slope_updater)
        self.play(FadeOut(slope_line), FadeOut(current_dot))
        pointed_x.set_value(4)
        slope_updater(slope_line)
        self.play(FadeIn(current_dot), FadeIn(slope_line))
        slope_line.add_updater(slope_updater)
        self.next_slide()

        ## Slide: descent, from left
        self.play(pointed_x.animate.set_value(local_minimum_1))
        self.next_slide()

        ## Slide: new point, in the small valley
        # slope_line.remove_updater(slope_updater)
        # self.play(FadeOut(slope_line), FadeOut(current_dot))
        # pointed_x.set_value(9)
        # slope_updater(slope_line)
        # self.play(FadeIn(current_dot), FadeIn(slope_line))
        # slope_line.add_updater(slope_updater)
        # self.next_slide()

        # ## Slide: descent, in the small valley
        # self.play(pointed_x.animate.set_value(local_minimum_shallow))
        # self.move_camera(frame_center=4 * RIGHT)
        # self.next_slide()

        ## Slide: gradient descent
        gradient_descent_text = Tex(r'{{ Gradient }} Descent', font_size=body_font_size)
        gradient_descent_text.next_to(minima_hard_text, DOWN, aligned_edge=LEFT)
        self.play(Write(gradient_descent_text))
        self.next_slide()

        ## Slide: gradient
        gradient_text = Tex(r'{{ Gradient }} = Slopes on drugs', font_size=body_font_size)
        gradient_text.next_to(minima_hard_text, DOWN, aligned_edge=LEFT)
        self.play(TransformMatchingTex(gradient_descent_text, gradient_text, False))
        self.next_slide()


class BackProp(ThreeDSlide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations

        body_font_size = 30
        slope_color = YELLOW
        network_input_color = WHITE
        network_output_color = PURPLE

        # x, y pairs
        f_pairwise = ((-2, 2), (1, 4), (5, 3), (6, 1), (9, 5), (10, 4), (12, 5),
                      (15, 3,), (15.5, 3.5), (16, 2), (17, 0.5),
                      (19, 4), (21, 5), (22, 10), (27, 0), (27.5, 1), (28, -0.5), (30, 3),
                      (33, -2),
                      (1_000_000, -100))
        x_points, f_values = zip(*f_pairwise)

        # Create a cubic spline interpolator
        f = scipy.interpolate.InterpolatedUnivariateSpline(x_points, f_values)
        f_d = f.derivative()
        local_minimum_1 = scipy.optimize.minimize_scalar(f, bounds=(6, 6.5), method='bounded').x

        ## Slide: title
        title = Text('Effective Training:\nInformation Flow')
        self.add(title)
        static_slide(self)
        self.next_slide()

        ## Slide: some points
        train_as_minim = Text('Training the network is a minimization problem')
        minim_with_gd = Text('We can minimize with Gradient Descent')
        gradients_are_hard = Text('In the wild, getting gradients (slopes) can be hard')
        body_group = VGroup(train_as_minim, minim_with_gd, gradients_are_hard)
        body_group.arrange(2 * DOWN)
        for text in body_group:
            text.font_size = body_font_size
            text.to_edge(LEFT)

        self.play(AnimationGroup(title.animate.to_edge(UP), Write(body_group), lag_ratio=0.6))
        self.next_slide()

        ## Slide: get the network on screen
        perceptron = Circle(0.2)
        perceptron_2 = Circle(0.2).shift(DOWN)
        perceptron_3 = Circle(0.2).shift(UP)
        perceptron_2_1 = Circle(0.2).shift(2 * RIGHT)
        perceptron_2_2 = Circle(0.2).shift(DR + RIGHT)
        perceptron_2_3 = Circle(0.2).shift(UR + RIGHT)
        layer_1 = [perceptron, perceptron_2, perceptron_3]
        layer_2 = [perceptron_2_3, perceptron_2_1, perceptron_2_2]

        x_label = MathTex('x').next_to(perceptron, LEFT).shift(UL)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT)
        z_label = MathTex('z').next_to(perceptron, LEFT).shift(DL)
        input_labels = [x_label, y_label, z_label]
        output_label = MathTex(r'o_1')
        output_label_2 = MathTex(r'o_2')
        output_label_3 = MathTex(r'o_3').shift(DOWN)
        output_labels = [output_label, output_label_2, output_label_3]

        for label in (output_label, output_label_2, output_label_3):
            label.font_size = body_font_size
            label.next_to(perceptron_2_1, RIGHT).shift(RIGHT)
        output_label.shift(UP)
        output_label_3.shift(DOWN)

        x_y_lines = (
            all_arrows((x_label, y_label, z_label), (perceptron, perceptron_2, perceptron_3))
            + all_arrows((perceptron, perceptron_2, perceptron_3),
                         (perceptron_2_3, perceptron_2_2, perceptron_2_1))
        )
        output_lines = pair_arrows((perceptron_2_1, perceptron_2_2, perceptron_2_3),
                                   (output_label_2, output_label_3, output_label))
        perceptron_group_multi = VGroup(perceptron, perceptron_2, perceptron_3,
                                        perceptron_2_1, perceptron_2_2, perceptron_2_3,
                                        x_label, y_label, z_label,
                                        *x_y_lines, output_label, output_label_2, output_label_3,
                                        *output_lines).move_to(UP)

        new_title = Text('Forward Propagation').to_edge(UP)

        # Axes for later demonstration
        ax_x_length = 10
        ax_y_length = 3
        ax_ratio = ax_y_length / ax_x_length        # Assuming they cover the same range
        ax = Axes((-1, 10), (-1, 10), x_length=ax_x_length, y_length=ax_y_length,
                  x_axis_config={'include_ticks': False},
                  y_axis_config={'include_ticks': False}).to_edge(DOWN, buff=0.1)
        function_plot = ax.plot(f, color=network_output_color)          # Will be useful later
        error_label = ax.get_y_axis_label('error')

        self.play(FadeOut(body_group))
        self.play(title.animate.become(new_title), Write(perceptron_group_multi),
                  Write(ax), Write(error_label))
        self.next_slide()

        ## Slide: forward flow
        def dots_for_transition(from_: list[VMobject], to: list[VMobject],
                                color: ParsableManimColor = WHITE):
            """Create all (pairwise) dots necessary to show a forward pass."""
            info_dots = [Dot(color=color).move_to(label) for _ in to for label in from_]
            dots_targets = [target for target in to for source in from_]
            return info_dots, dots_targets

        def dots_for_output_transition(from_: list[VMobject], to: list[VMobject],
                                       color: ParsableManimColor = WHITE):
            """Create all (paired) dots necessary to show the output pass."""
            info_dots = [Dot(color=color).move_to(label) for label in from_]
            return info_dots, to

        # Do a couple of forward passes and plot the curve
        forward_x_values = (1, 4, 7, 8)
        on_plot_outputs_group = VGroup()
        l2_color = network_input_color.interpolate(network_output_color, 0.5)
        for x_value in forward_x_values:
            l1_dots, l1_targets = dots_for_transition(input_labels, layer_1, network_input_color)
            self.play(*(Create(info_dot) for info_dot in l1_dots))
            self.play(*(info_dot.animate.become(info_dot.copy().move_to(target)
                                                .set_color(l2_color))
                        for info_dot, target in zip(l1_dots, l1_targets)))

            self.remove(*l1_dots)
            l2_dots, l2_targets = dots_for_transition(layer_1, layer_2, l2_color)
            self.add(*l2_dots)
            self.play(*(info_dot.animate.become(info_dot.copy().move_to(target)
                                                        .set_color(network_output_color))
                        for info_dot, target in zip(l2_dots, l2_targets)))

            self.remove(*l2_dots)
            out_dots, out_targets = dots_for_output_transition(layer_2, output_labels,
                                                               network_output_color)
            self.add(*out_dots)
            self.play(*(info_dot.animate.move_to(target)
                        for info_dot, target in zip(out_dots, out_targets)))

            # Output goes to the plot
            output_group = VGroup(*out_dots)
            new_on_plot_dot = Dot(ax.i2gp(x_value, function_plot), color=network_output_color)
            on_plot_outputs_group.add(new_on_plot_dot)
            self.play(ReplacementTransform(output_group, new_on_plot_dot))

        self.play(FadeOut(on_plot_outputs_group), Write(function_plot))

        self.next_slide()

        ## Slide: backward flow
        new_title = Text('Backpropagation').to_edge(UP)
        self.play(Transform(title, new_title))

        backward_x_values = np.linspace(3, local_minimum_1, 4)
        for x_value in backward_x_values:
            # Dot chosen from the plot becomes the new backprop input
            plot_dot = Dot(ax.i2gp(x_value, function_plot), color=network_output_color)
            out_dots, out_targets = dots_for_output_transition(output_labels, layer_2,
                                                               network_output_color)
            backprop_input_group = VGroup(*out_dots)

            self.play(Create(plot_dot))
            self.play(ReplacementTransform(plot_dot, backprop_input_group))
            self.play(*(info_dot.animate.move_to(target)
                        for info_dot, target in zip(out_dots, out_targets)))

            self.remove(*out_dots)
            l2_dots, l2_targets = dots_for_transition(layer_2, layer_1, network_output_color)
            l1_color = network_output_color.interpolate(slope_color, 0.5)
            self.add(*l2_dots)
            self.play(*(info_dot.animate.become(info_dot.copy().move_to(target)
                                                .set_color(l1_color))
                        for info_dot, target in zip(l2_dots, l2_targets)))

            self.remove(*l2_dots)
            l1_dots, l1_targets = dots_for_transition(layer_1, input_labels, l1_color)
            self.add(*l1_dots)
            self.play(*(info_dot.animate.become(info_dot.copy().move_to(target)
                                                .set_color(slope_color))
                        for info_dot, target in zip(l1_dots, l1_targets)))
            backprop_output_group = VGroup(*l1_dots)

            # Finally, get the slope
            slope_angle = np.arctan(f_d(x_value) * ax_ratio)
            slope_line = (Line(ORIGIN, np.array([np.cos(slope_angle), np.sin(slope_angle), 0.]),
                               color=slope_color)
                          .move_to(ax.i2gp(x_value, function_plot)))

            self.play(ReplacementTransform(backprop_output_group, slope_line))

        self.next_slide()
