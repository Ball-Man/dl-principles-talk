from itertools import chain, product
from collections.abc import Iterable

import numpy as np
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


def all_arrows(from_objects: Iterable[VMobject], to_objects: Iterable[VMobject],
               line_factory=Arrow) -> list[VMobject]:
    """Get lines connecting the objects in from_ to the objects in to_."""
    lines = []
    for from_, to in product(from_objects, to_objects):
        lines.append(line_factory(from_.get_critical_point(RIGHT), to.get_critical_point(LEFT)))
    return lines


class LinearToNonLinear(Slide):

    def construct(self):
        self.wait_time_between_slides = 0.1      # Fix incomplete animations
        formula_font_size = 40

        ## Slide: title
        title = Text('Logistic Regressor')
        self.add(title)

        static_slide(self)
        self.next_slide()

        ## Slide: Function definition
        function_def = MathTex(r'f: \mathbb{R}^2 \to \mathbb{R}').shift(UP)
        function_def.font_size = formula_font_size
        function_def.to_edge(LEFT)

        regressor_formula = MathTex(r'f(x, y) = \sigma(w_1x + w_2y + b)',
                                    substrings_to_isolate=[r'f(x, y) = \sigma(w_1x + w_2y', ')'])
        regressor_formula.font_size = formula_font_size
        regressor_formula.to_edge(LEFT)

        self.play(title.animate.to_edge(UP), Write(function_def), Write(regressor_formula))

        self.next_slide()

        ## Slide: get rid of b
        regressor_formula_no_b = MathTex(r'f(x, y) = \sigma(w_1x + w_2y)',
                                         substrings_to_isolate=[r'f(x, y) = \sigma(w_1x + w_2y', ')'])
        regressor_formula_no_b.font_size = formula_font_size
        regressor_formula_no_b.to_edge(LEFT)

        self.play(TransformMatchingTex(regressor_formula, regressor_formula_no_b))
        regressor_formula = regressor_formula_no_b

        self.next_slide()

        ## Slide: matrix form
        regressor_matrix_formula = MathTex(
            r'''
            f \left( \left[ \begin{array}{c} x \\ y \end{array} \right] \right) =
            \sigma\left(
            \left[ \begin{array}{cc}
            w_1 & w_2
            \end{array} \right]
            \left[ \begin{array}{c}
            x \\
            y
            \end{array} \right]
            \right)
            '''
        )
        regressor_matrix_formula.font_size = formula_font_size
        regressor_matrix_formula.to_edge(LEFT)

        # Animate the expression
        self.play(TransformMatchingShapes(regressor_formula, regressor_matrix_formula))
        self.next_slide()

        ## Slide: the perceptron
        perceptron = Circle(0.2)
        x_label = MathTex('x').next_to(perceptron, LEFT).shift(LEFT + 0.5 * UP)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT + 0.5 * DOWN)
        output_label = MathTex(
            r'f \left( \left[ \begin{array}{c} x \\ y \end{array} \right] \right)'
        )
        output_label.font_size = formula_font_size
        output_label.next_to(perceptron, RIGHT).shift(RIGHT)
        x_y_lines = all_arrows((x_label, y_label), (perceptron,))
        output_line = Arrow(perceptron.get_critical_point(RIGHT),
                            output_label.get_critical_point(LEFT))
        perceptron_group = VGroup(perceptron, x_label, y_label, *x_y_lines,
                                  output_label, output_line).shift(RIGHT * 3)

        self.play(Create(perceptron_group, lag_ratio=0))
        self.next_slide()

        ## Slide: multivaried function
        # Update the matrix form
        regressor_matrix_formula_multi = MathTex(
            r'''
            f \left( \left[ \begin{array}{c} x \\ y \\ z \end{array} \right] \right) =
            \sigma\left(
            \left[ \begin{array}{ccc}
            w_1 & w_2 & w_3
            \end{array} \right]
            \left[ \begin{array}{c}
            x \\
            y \\
            z
            \end{array} \right]
            \right)
            '''
        )
        regressor_matrix_formula_multi.font_size = formula_font_size
        regressor_matrix_formula_multi.to_edge(LEFT).shift(0.5 * DOWN)

        # Update function def
        function_def_multi = MathTex(r'f: \mathbb{R}^3 \to \mathbb{R}').shift(UP)
        function_def_multi.font_size = formula_font_size
        function_def_multi.to_edge(LEFT)

        # Update perceptron
        perceptron = Circle(0.2)
        x_label = MathTex('x').next_to(perceptron, LEFT).shift(UL)
        y_label = MathTex('y').next_to(perceptron, LEFT).shift(LEFT)
        z_label = MathTex('z').next_to(perceptron, LEFT).shift(DL)
        output_label = MathTex(
            r'f \left( \left[ \begin{array}{c} x \\ y \\ z\end{array} \right] \right)'
        )
        output_label.font_size = formula_font_size
        output_label.next_to(perceptron, RIGHT).shift(RIGHT)
        x_y_lines = all_arrows((x_label, y_label, z_label), (perceptron,))
        output_line = Arrow(perceptron.get_critical_point(RIGHT),
                            output_label.get_critical_point(LEFT))
        perceptron_group_multi = VGroup(perceptron, x_label, y_label, z_label, *x_y_lines,
                                        output_label, output_line).shift(RIGHT * 3)

        self.play(TransformMatchingShapes(regressor_matrix_formula,
                                          regressor_matrix_formula_multi),
                  TransformMatchingShapes(function_def, function_def_multi),
                  FadeTransform(perceptron_group, perceptron_group_multi))
        self.next_slide()
