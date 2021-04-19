import numpy as np
import torch
from typing import List
from utils.generators.mixed_len_generator import Parser, SimulateStack
from utils.train_utils import validity

class CSGEngine:
    def __init__(self, unique_draws: List, steps: int, canvas_shape: List):
        """
        This class parses complete output from the network which are in joint
        fashion. This class can be used to generate final canvas and
        expressions.
        :param unique_draws: Unique draw/op operations in the current dataset
        :param stack_size: Stack size
        :param steps: Number of steps in the program
        :param canvas_shape: Shape of the canvases
        """
        self.canvas_shape = canvas_shape
        self.stack_size = steps // 2 + 1
        self.steps = steps
        self.Parser = Parser()
        self.sim = SimulateStack(self.stack_size, self.canvas_shape)
        self.unique_draws = unique_draws

    def get_final_canvas(self,
                         outputs: List,
                         if_just_expressions=False,
                         if_pred_images=False):
        """
        Takes the raw output from the network and returns the predicted
        canvas. The steps involve parsing the outputs into expressions,
        decoding expressions, and finally producing the canvas using
        intermediate stacks.
        :param if_just_expressions: If only expression is required than we
        just return the function after calculating expressions
        :param outputs: List, each element correspond to the output from the
        network
        :return: stack: Predicted final stack for correct programs
        :return: correct_programs: Indices of correct programs
        """
        batch_size = outputs[0].size()[0]

        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        labels = [torch.max(outputs[i], 1)[1].data.cpu().numpy() for i in range(self.steps)]
        for j in range(batch_size):
            for i in range(self.steps):
                expressions[j] += self.unique_draws[labels[i][j]]

        # Remove the stop symbol and later part of the expression
        for index, exp in enumerate(expressions):
            expressions[index] = exp.split("$")[0]
        if if_just_expressions:
            return expressions
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.Parser.parse(exp)
            if validity(program, len(program), len(program) - 1):
                correct_programs.append(index)
            else:
                if if_pred_images:
                    # if you just want final predicted image
                    stack = np.zeros((self.canvas_shape[0],
                                      self.canvas_shape[1]))
                else:
                    stack = np.zeros(
                        (self.steps + 1, self.stack_size, self.canvas_shape[0],
                         self.canvas_shape[1]))
                stacks.append(stack)
                continue
                # Check the validity of the expressions

            self.sim.generate_stack(program)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            if if_pred_images:
                stacks.append(stack[-1, 0, :, :])
            else:
                stacks.append(stack)
        if len(stacks) == 0:
            return None
        if if_pred_images:
            stacks = np.stack(stacks, 0).astype(dtype=np.bool)
        else:
            stacks = np.stack(stacks, 1).astype(dtype=np.bool)
        return stacks, correct_programs, expressions

    def expression2stack(self, expressions: List):
        """Assuming all the expression are correct and coming from
        groundtruth labels. Helpful in visualization of programs
        :param expressions: List, each element an expression of program
        """
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.Parser.parse(exp)
            self.sim.generate_stack(program)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            stacks.append(stack)
        stacks = np.stack(stacks, 1).astype(dtype=np.float32)
        return stacks

    def labels2exps(self, labels: np.ndarray, steps: int):
        """
        Assuming grountruth labels, we want to find expressions for them
        :param labels: Grounth labels batch_size x time_steps
        :return: expressions: Expressions corresponding to labels
        """
        if isinstance(labels, np.ndarray):
            batch_size = labels.shape[0]
        else:
            batch_size = labels.size()[0]
            labels = labels.data.cpu().numpy()
        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        for j in range(batch_size):
            for i in range(steps):
                expressions[j] += self.unique_draws[labels[j, i]]
        return expressions
