import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet

class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init. Be aware that it gets called only once
        and then the solve method will get called several times.
        """
        pass

    def determineColor(self, cell):
        if cell == 0:
            return "0Black"
        elif cell == 1:
            return "1Blue"
        elif cell == 2:
            return "2Red"
        elif cell == 5:
            return "5Gray"

    def appendCell(self, cellCoords, grid, res):
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        cellx = cellCoords[0]
        celly = cellCoords[1]
        if cellx >= 0 and cellx < numRows and celly >= 0 and celly < numCols:
            res.append((cellx, celly))
        return res

    # recursive, finds all of the cells that are adjacent to a certain cell
    def findAdjacentCells(self, cellCoords, celly, grid, adjacentCells):
        cellUpLeft = (cellx - 1, celly - 1)
        cellUp = (cellx, celly - 1)
        cellUpRight = (cellx + 1, celly + 1)
        cellRight = (cellx + 1, celly)
        cellDownRight = (cellx - 1, celly + 1)
        cellDown = (cellx, celly - 1)
        cellDownLeft = (cellx - 1, celly - 1)
        cellLeft = (cellx - 1, celly)
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        emptyList = []
        resUpLeft = self.appendCell(cellUpLeft, grid, emptyList)
        if resUpLeft != emptyList:
            return self.findAdjacentCells(cellUpLeft, grid, adjacentCells)
        resUp = self.appendCell(cellUp, grid, emptyList)
        if resUp != emptyList:
            return self.findAdjacentCells(cellUp, grid, adjacentCells)
        resUpRight = self.appendCell(cellUpRight, grid, emptyList)
        if resUpRight != emptyList:
            return self.findAdjacentCells(cellUpRight, grid, adjacentCells)
        resRight = self.appendCell(cellRight, grid, emptyList)
        if resRight != emptyList:
            return self.findAdjacentCells(cellRight, grid, adjacentCells)
        resDownRight = self.appendCell(cellDownRight, grid, emptyList)
        if resDownRight != emptyList:
            return self.findAdjacentCells(cellDownRight, grid, adjacentCells)
        resDown = self.appendCell(cellDown, grid, emptyList)
        if resDown != emptyList:
            return self.findAdjacentCells(cellDown, grid, adjacentCells)
        resDownLeft = self.appendCell(cellDownLeft, grid, emptyList)
        if resDownLeft != emptyList:
            return self.findAdjacentCells(cellDownLeft, grid, adjacentCells)
        return adjacentCells
        # return self.findAdjacentCells(cellCoords, celly, grid, adjacentCells)
        """if cellUpLeft[0] >= 0 and cellUpLeft[0] < numRows and cellUpLeft[1] >= 0 and cellUpLeft[1] < numCols:
            adjacentCells.append((cellUpLeft))
            return self.findAdjacentCells(cell, cellx - 1, celly - 1, grid, adjacentCells)
        if cellUp[0] >= 0 and cellUp[0] < numRows and cellUp[1] >= 0 and cellUp[1] < numCols:
            adjacentCells.append((cellUpLeft))
            return self.findAdjacentCells(cell, cellx, celly, grid, adjacentCells)"""

    def findShapes(self, grid):
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        i = 0
        shapesList = []
        adjacentCellsList = []
        adjacentCellsVisited = []
        while i < numRows:
            j = 0
            while j < numCols:
                cellCurr = grid[i, j]
                cellCurrx = i
                cellCurry = j
                cellCoords = (cellCurrx, cellCurry)
                self.findAdjacentCells(cellCurr, cellCoords, grid, adjacentCellsList, adjacentCellsVisited)

                j += 1
            i += 1

    def findBackgroundColor(self, grid):

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method
        to solve the incoming ArcProblem.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer.

        Also, you shouldn't add more than 3 predictions to the list as
        that is considered an ERROR and the test will be automatically
        marked as incorrect.
        """

        findLinked(self, )
        # 0520fde7
        if

        # return np.array([[[2, 0, 2], [0, 0, 0], [0, 0, 0]]])

        predictions: list[np.ndarray] = list()

        '''
        The next 2 lines are only an example of how to populate the predictions list.
        This will just be an empty answer the size of the input data;
        delete it before you start adding your own predictions.
        '''
        output = np.zeros_like(arc_problem.test_set().get_input_data().data())
        predictions.append(output)
        return
        # return predictions
