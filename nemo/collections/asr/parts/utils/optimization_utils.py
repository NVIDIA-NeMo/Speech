# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The original code of Linear Sum Assignment solver is
# from: https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py
# The following is the full text of the license:

# Hungarian algorithm (Kuhn-Munkres) for solving the linear sum assignment
# problem. Taken from scikit-learn. Based on original code by Brian Clapper,
# adapted to NumPy by Gael Varoquaux.
# Further improvements by Ben Root, Vlad Niculae and Lars Buitinck.
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD

import torch


@torch.jit.script
def unravel_index(index: int, shape: torch.Tensor):
    """
    Unravel the index input to fit the given shape.
    This function is needed for torch.jit.script compatibility.

    Args:
        index (int): The index to unravel.
        shape (Tesnor): The shape to unravel the index to.

    Returns:
        Tensor: The unraveled index.
    """
    out = []
    shape = torch.flip(shape, dims=(0,))
    for dim in shape:
        out.append(index % dim)
        index = index // dim
    out = torch.tensor([int(x.item()) for x in out])
    return torch.flip(out, dims=(0,))


@torch.jit.script
class LinearSumAssignmentSolver(object):
    """
    A Solver class for the linear sum assignment (LSA) problem. 
    Designed for torch.jit.script compatibility in NeMo. 
        
    The LSA problem is also referred to as bipartite matching problem. An LSA problem is described 
    by a matrix `cost_mat`, where each cost_mat[i,j] is the cost of matching vertex i of the first partite 
    set (e.g. a "worker") and vertex j of the second set (e.g. a "job"). 
    
    Thus, the goal of LSA-solver is to find a complete assignment of column element to row element with 
    the minimal cost. Note that the solution may not be unique and there could be multiple solutions that 
    yield the same minimal cost.

    LSA problem solver is needed for the following tasks in NeMo: 
        - Permutation Invariant Loss (PIL) for diarization model training
        - Label permutation matching for online speaker diarzation 
        - Concatenated minimum-permutation Word Error Rate (cp-WER) calculation 

    This implementation is based on the LAP solver from scipy: 
        https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py
        The scipy implementation comes with the following license:
    
        Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
        Author: Brian M. Clapper, Gael Varoquaux
        License: 3-clause BSD

    References
        1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
        2. https://en.wikipedia.org/wiki/Hungarian_algorithm
        3. https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py


    Attributes:
        cost_mat (Tensor): 2D matrix containing cost matrix. Number of columns must be larger than number of rows.
        row_uncovered (Tensor): 1D matrix containing boolean values indicating whether a row is covered.
        col_uncovered (Tensor): 1D matrix containing boolean values indicating whether a column is covered.
        zero_row (Tensor): 1D matrix containing the row index of the last zero found.
        zero_col (Tensor): 1D matrix containing the column index of the last zero found.
        path (Tensor): 2D matrix containing the path taken through the matrix.
        marked (Tensor): 2D matrix containing the marked zeros.
    """

    def __init__(self, cost_matrix: torch.Tensor):
        # The main cost matrix
        self.cost_mat = cost_matrix
        row_len, col_len = self.cost_mat.shape

        # Initialize the solver state
        self.zero_row = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)
        self.zero_col = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)

        # Initialize the covered matrices
        self.row_uncovered = torch.ones(row_len, dtype=torch.bool).to(cost_matrix.device)
        self.col_uncovered = torch.ones(col_len, dtype=torch.bool).to(cost_matrix.device)

        # Initialize the path matrix and the mark matrix
        self.path = torch.zeros((row_len + col_len, 2), dtype=torch.long).to(cost_matrix.device)
        self.marked = torch.zeros((row_len, col_len), dtype=torch.long).to(cost_matrix.device)

    def _reset_uncovered_mat(self):
        """
        Clear all covered matrix cells and assign `True` to all uncovered elements.
        """
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True

    def _step1(self):
        """
        Step 1

        Goal: Subtract the smallest element of each row from its elements.
            - All elements of the matrix are now non-negative.
            - Therefore, an assignment of total cost 0 is the minimum cost assignment.
            - This operation leads to at least one zero in each row.

        Procedure:
        - For each row of the matrix, find the smallest element and subtract it from every element in its row.
        - Go to Step 2.
        """
        self.cost_mat -= torch.min(self.cost_mat, dim=1)[0].unsqueeze(1)
        return 2

    def _step2(self):
        """
        Step 2

        Goal: Make sure assignment with cost sum 0 is feasible.

        Procedure:
        - Find a zero in the resulting cost matrix. 
        - If there are no marked zeros in its row or column, mark the zero. 
        - Repeat for each element in the matrix.
        - Go to step 3.
        """
        ind_out = torch.where(self.cost_mat == 0)
        ind, val = list(ind_out[0]), list(ind_out[1])
        for i, j in zip(ind, val):
            if self.col_uncovered[j] and self.row_uncovered[i]:
                self.marked[i, j] = 1
                self.col_uncovered[j] = False
                self.row_uncovered[i] = False

        self._reset_uncovered_mat()
        return 3

    def _step3(self) -> int:
        """
        Step 3
        
        Goal: All zeros in the matrix must be covered by marking with the least numbers of rows and columns.

        Procedure:
            - Cover each column containing a marked zero. 
                - If n columns are covered, the marked zeros describe a complete set of unique assignments.
                In this case, Go to Step 0 (Done state)
                - Otherwise, Go to Step 4.
        """
        marked = self.marked == 1
        self.col_uncovered[torch.any(marked, dim=0)] = False
        if marked.sum() < self.cost_mat.shape[0]:
            return 4  # Go to step 4
        else:
            return 0  # Go to step 0 (Done state)

    def _step4(self, bypass: bool = False) -> int:
        """
        Step 4

        Goal: Cover all columns containing a marked zero.

        Procedure:
        - Find a non-covered zero and put a prime mark on it. 
            - If there is no marked zero in the row containing this primed zero, Go to Step 5.
            - Otherwise, cover this row and uncover the column containing the marked zero. 
        - Continue in this manner until there are no uncovered zeros left. 
        - Save the smallest uncovered value.
        - Go to Step 6.
        """
        # We convert to int as numpy operations are faster on int
        cost_mat = (self.cost_mat == 0).int()
        covered_cost_mat = cost_mat * self.row_uncovered.unsqueeze(1)
        covered_cost_mat *= self.col_uncovered.long()
        row_len, col_len = self.cost_mat.shape
        if not bypass:
            while True:
                urv = unravel_index(torch.argmax(covered_cost_mat).item(), torch.tensor([col_len, row_len]))
                row, col = int(urv[0].item()), int(urv[1].item())
                if covered_cost_mat[row, col] == 0:
                    return 6
                else:
                    self.marked[row, col] = 2  # Find the first marked element in the row
                    mark_col = torch.argmax((self.marked[row] == 1).int())
                    if self.marked[row, mark_col] != 1:  # No marked element in the row
                        self.zero_row = torch.tensor(row)
                        self.zero_col = torch.tensor(col)
                        return 5
                    else:
                        col = mark_col
                        self.row_uncovered[row] = False
                        self.col_uncovered[col] = True
                        covered_cost_mat[:, col] = cost_mat[:, col] * self.row_uncovered
                        covered_cost_mat[row] = 0
        return 0

    def _step5(self) -> int:
        """
        Step 5

        Goal: Construct a series of alternating primed and marked zeros as follows.
        
        Procedure:
        - Let Z0 represent the uncovered primed zero found in Step 4.
        - Let Z1 denote the marked zero in the column of Z0 (if any).
        - Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        - Continue until the series terminates at a primed zero that has no marked zero in its column. 
        - Unmark each marked zero of the series.
        - Mark each primed zero of the series.
        - Erase all primes and uncover every line in the matrix. 
        - Return to Step 3
        """
        count = torch.tensor(0)
        path = self.path
        path[count, 0] = self.zero_row.long()
        path[count, 1] = self.zero_col.long()

        while True:  # Unmark each marked zero of the series
            # Find the first marked element in the col defined by the path (= `val`)
            row = torch.argmax((self.marked[:, path[count, 1]] == 1).int())

            if self.marked[row, path[count, 1]] != 1:
                # Could not find one
                break
            else:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count - 1, 1]

            # Find the first prime element in the row defined by the first path step
            col = int(torch.argmax((self.marked[path[count, 0]] == 2).int()))
            if self.marked[row, col] != 2:
                col = -1
            count += 1
            path[count, 0] = path[count - 1, 0]
            path[count, 1] = col

        # Convert paths
        for i in range(int(count.item()) + 1):
            if self.marked[path[i, 0], path[i, 1]] == 1:
                self.marked[path[i, 0], path[i, 1]] = 0
            else:
                self.marked[path[i, 0], path[i, 1]] = 1

        self._reset_uncovered_mat()

        # Remove all prime markings in marked matrix
        self.marked[self.marked == 2] = 0
        return 3

    def _step6(self) -> int:
        """
        Step 6

        Goal: Prepare for another iteration by modifying the cost matrix.

        Procedure:
        - Add the value found in Step 4 to every element of each covered row.
        - Subtract it from every element of each uncovered column.
        - Return to Step 4 without altering any marks, primes, or covered lines.
        """
        if torch.any(self.row_uncovered) and torch.any(self.col_uncovered):
            row_minval = torch.min(self.cost_mat[self.row_uncovered], dim=0)[0]
            minval = torch.min(row_minval[self.col_uncovered])
            self.cost_mat[~self.row_uncovered] += minval
            self.cost_mat[:, self.col_uncovered] -= minval
        return 4


@torch.jit.script
def linear_sum_assignment(cost_matrix: torch.Tensor, max_size: int = 100):
    """
    Launch the linear sum assignment algorithm on a cost matrix.

    Args:
        cost_matrix (Tensor): The cost matrix of shape (N, M) where M should be larger than N.

    Returns:
        row_index (Tensor): The row indices of the optimal assignments.
        col_index (Tensor): The column indices of the optimal assignments.
    """
    cost_matrix = cost_matrix.clone().detach()

    if len(cost_matrix.shape) != 2:
        raise ValueError(f"2-d tensor is expected but got a {cost_matrix.shape} tensor")
    if max(cost_matrix.shape) > max_size:
        raise ValueError(
            f"Cost matrix size {cost_matrix.shape} is too large. The maximum supported size is {max_size}x{max_size}."
        )

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    lap_solver = LinearSumAssignmentSolver(cost_matrix)
    f_int: int = 0 if 0 in cost_matrix.shape else 1

    # while step is not Done (step 0):
    # NOTE: torch.jit.scipt does not support getattr with string argument.
    # Do not use getattr(lap_solver, f"_step{f_int}")()
    while f_int != 0:
        if f_int == 1:
            f_int = lap_solver._step1()
        elif f_int == 2:
            f_int = lap_solver._step2()
        elif f_int == 3:
            f_int = lap_solver._step3()
        elif f_int == 4:
            f_int = lap_solver._step4()
        elif f_int == 5:
            f_int = lap_solver._step5()
        elif f_int == 6:
            f_int = lap_solver._step6()

    if transposed:
        marked = lap_solver.marked.T
    else:
        marked = lap_solver.marked
    row_index, col_index = torch.where(marked == 1)
    return row_index, col_index
