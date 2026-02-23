import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from copy import deepcopy

from collections import deque

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

    def determineBackgroundColor(self, grid):
        gridFlattened = grid.flatten()
        mostCommonColorBincount = np.bincount(gridFlattened)
        mostCommonColor = np.argmax(mostCommonColorBincount)
        return mostCommonColor

    # determines whether a cell can be included or if it is off the grid
    def isValidCell(self, cellCoords, grid, res):
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        cellx = cellCoords[0]
        celly = cellCoords[1]
        result = False
        if cellx >= 0 and cellx < numRows and celly >= 0 and celly < numCols:
            # res.append((cellx, celly))
            result = True
        return result

    def isNotBackgroundColor(self, cellCoords, backgroundColor, grid):
        cellx = cellCoords[0]
        celly = cellCoords[1]
        currCell = grid[cellx, celly]
        if currCell == backgroundColor:
            return False
        else:
            return True

    def isSameColor(self, cellCoords, color, grid):
        cellx = cellCoords[0]
        celly = cellCoords[1]
        currCell = grid[cellx, celly]
        if currCell == color:
            return True
        else:
            return False

    # finds all of the cells that are adjacent to a given cell that are part of the same shape
    def findAdjacentCells(self, cellCoords, grid):
        # based on my Mini Project 1 implementation
        setPrevVisited = set()
        dictPredecessors = dict()
        dictEmpty = dict()
        dequeCurr = deque()
        cellx = cellCoords[0]
        celly = cellCoords[1]
        cellCurr = grid[cellx, celly]
        backgroundColor = self.determineBackgroundColor(grid)
        # something like find all te adjacent cells that are not the background color
        dequeEmpty = deque()
        dequeCurr.append(cellCoords)
        adjacentCellsList = []
        emptyList = []
        setPrevVisited.add(cellCoords)
        while dequeCurr != dequeEmpty:
            currCellCoords = dequeCurr.popleft()
            cellx = currCellCoords[0]
            celly = currCellCoords[1]
            cellUpLeft = (cellx - 1, celly - 1)
            cellUp = (cellx, celly - 1)
            cellUpRight = (cellx + 1, celly - 1)
            cellRight = (cellx + 1, celly)
            cellDownRight = (cellx + 1, celly + 1)
            cellDown = (cellx, celly + 1)
            cellDownLeft = (cellx - 1, celly + 1)
            cellLeft = (cellx - 1, celly)
            if dictPredecessors == dictEmpty:
                dictPredecessors[currCellCoords] = None
            cellColorCurr = grid[cellx, celly]
            dictPredecessors[currCellCoords] = currCellCoords # don't think i actually need this for this one
            adjacentCellsList.append(currCellCoords)
            resUpLeft = self.isValidCell(cellUpLeft, grid, emptyList)
            if resUpLeft == True:
                if self.isSameColor(cellUpLeft, cellCurr, grid) and cellUpLeft not in setPrevVisited:
                    dequeCurr.append(cellUpLeft)
                    setPrevVisited.add(cellUpLeft)
            resUp = self.isValidCell(cellUp, grid, emptyList)
            if resUp == True:
                if self.isSameColor(cellUp, cellCurr, grid) and cellUp not in setPrevVisited:
                    dequeCurr.append(cellUp)
                    setPrevVisited.add(cellUp)
            resUpRight = self.isValidCell(cellUpRight, grid, emptyList)
            if resUpRight == True:
                if self.isSameColor(cellUpRight, cellCurr, grid) and cellUpRight not in setPrevVisited:
                    dequeCurr.append(cellUpRight)
                    setPrevVisited.add(cellUpRight)
            resRight = self.isValidCell(cellRight, grid, emptyList)
            if resRight == True:
                if self.isSameColor(cellRight, cellCurr, grid) and cellRight not in setPrevVisited:
                    dequeCurr.append(cellRight)
                    setPrevVisited.add(cellRight)
            resDownRight = self.isValidCell(cellDownRight, grid, emptyList)
            if resDownRight == True:
                if self.isSameColor(cellDownRight, cellCurr, grid) and cellDownRight not in setPrevVisited:
                    dequeCurr.append(cellDownRight)
                    setPrevVisited.add(cellDownRight)
            resDown = self.isValidCell(cellDown, grid, emptyList)
            if resDown == True:
                if self.isSameColor(cellDown, cellCurr, grid) and cellDown not in setPrevVisited:
                    dequeCurr.append(cellDown)
                    setPrevVisited.add(cellDown)
            resDownLeft = self.isValidCell(cellDownLeft, grid, emptyList)
            if resDownLeft == True:
                if self.isSameColor(cellDownLeft, cellCurr, grid) and cellDownLeft not in setPrevVisited:
                    dequeCurr.append(cellDownLeft)
                    setPrevVisited.add(cellDownLeft)
            resLeft = self.isValidCell(cellLeft, grid, emptyList)
            if resLeft == True:
                if self.isSameColor(cellLeft, cellCurr, grid) and cellLeft not in setPrevVisited:
                    dequeCurr.append(cellLeft)
                    setPrevVisited.add(cellLeft)
        return adjacentCellsList


    # def overlapBetweenShapes(self, ):

    # 0520fde7 now it's just for half the grid and all divided by cols but generalize this
    def findOverlappingCells(self, grid):
        gridShape = grid.shape
        gridNumRows = gridShape[0]
        gridNumCols = gridShape[1]
        gridMiddle = gridNumCols // 2
        gridFirstHalfLast = gridMiddle - 1
        gridLastHalfLast = gridMiddle
        backgroundColor = self.determineBackgroundColor(grid)
        i = 0
        j = 0
        result = np.zeros((gridNumRows, gridMiddle))
        while i < gridNumRows:
            j = 0
            while j < gridMiddle:
                gridFirstHalfCurr = grid[i][j]
                gridLastHalfCurr = grid[i][j + gridMiddle]
                if gridFirstHalfCurr == backgroundColor or gridLastHalfCurr == backgroundColor:
                    j += 1
                    continue
                # elif self.isSameColor(gridFirstHalfCurr, gridLastHalfCurr, gridMiddle):
                elif gridFirstHalfCurr == gridLastHalfCurr:
                    result[i, j] = gridFirstHalfCurr
                else:
                    result[i, j] = backgroundColor
                j += 1
            i += 1
        return result

    def markOverlappingCellsTraining(self, grid1, grid2, output):
        gridShape1 = grid1.shape
        gridShape2 = grid2.shape
        grid1NumRows = gridShape1[0]
        grid1NumCols = gridShape1[1]
        grid2NumRows = gridShape2[0]
        grid2NumCols = gridShape2[1]
        """backgroundColor1 = self.determineBackgroundColor(grid1)
        backgroundColor2 = self.determineBackgroundColor(grid2)"""
        backgroundColor1 = 0
        backgroundColor2 = 0 # not always true but not sure how to fix this and it's true in most csaes
        result = np.zeros((grid1NumRows, grid1NumCols))
        i = 0
        j = 0
        outputColor = 0
        while i < grid1NumRows:
            j = 0
            while j < grid1NumCols:
                grid1Curr = grid1[i, j]
                grid2Curr = grid2[i, j]
                if grid1Curr == backgroundColor1 or grid2Curr == backgroundColor2:
                    result[i, j] = backgroundColor1
                elif grid1Curr == grid2Curr:
                    outputColor = output[i, j]
                    result[i, j] = outputColor
                elif grid1Curr != grid2Curr:
                    result[i, j] = backgroundColor1
                j += 1
                # print("markoverlapping j ")
            i += 1
            # print("markoverlapping i ")
        return (result, outputColor)

    def markOverlappingCells(self, grid1, grid2, outputColor):
        # print(outputColor)
        gridShape1 = grid1.shape
        gridShape2 = grid2.shape
        grid1NumRows = gridShape1[0]
        grid1NumCols = gridShape1[1]
        grid2NumRows = gridShape2[0]
        grid2NumCols = gridShape2[1]
        backgroundColor1 = self.determineBackgroundColor(grid1)
        backgroundColor2 = self.determineBackgroundColor(grid2)
        backgroundColor1 = 0
        backgroundColor2 = 0 # not alwasy true but it's typically the case and is true for everything in b
        result = np.zeros((grid1NumRows, grid1NumCols))
        i = 0
        j = 0
        while i < grid1NumRows:
            j = 0
            while j < grid1NumCols:
                grid1Curr = grid1[i, j]
                grid2Curr = grid2[i, j]
                if grid1Curr == backgroundColor1 or grid2Curr == backgroundColor2:
                    result[i, j] = backgroundColor1
                elif grid1Curr == grid2Curr:
                    result[i, j] = outputColor
                elif grid1Curr != grid2Curr:
                    result[i, j] = backgroundColor1
                j += 1
                # print("markoverlapping j ")
            i += 1
            # print("markoverlapping i ")
        return result

    def markNonOverlappingCellsHorizontallyTraining(self, grid1, grid2, output):
        gridShape1 = grid1.shape
        gridShape2 = grid2.shape
        grid1NumRows = gridShape1[0]
        grid1NumCols = gridShape1[1]
        grid2NumRows = gridShape2[0]
        grid2NumCols = gridShape2[1]
        """backgroundColor1 = self.determineBackgroundColor(grid1)
        backgroundColor2 = self.determineBackgroundColor(grid2)"""
        backgroundColor1 = 0
        backgroundColor2 = 0 # not always true but not sure how to fix this and it's true in most csaes
        result = np.zeros((grid1NumRows, grid1NumCols))
        i = 0
        j = 0
        """print(grid1)
        print(grid2)"""
        outputColor = 0
        while i < grid1NumRows - 1:
            j = 0
            while j < grid1NumCols:
                grid1Curr = grid1[i, j]
                grid2Curr = grid2[i, j]
                if grid1Curr != backgroundColor1 or grid2Curr != backgroundColor2:
                    result[i, j] = backgroundColor1
                elif grid1Curr == backgroundColor1 and grid2Curr == backgroundColor2:
                    outputColor = output[i, j]
                    result[i, j] = outputColor
                j += 1
                # print("markoverlapping j ")
            i += 1
            # print("markoverlapping i ")
        return (result, outputColor)

    def markNonOverlappingCellsHorizontally(self, grid1, grid2, outputColor):
        gridShape1 = grid1.shape
        gridShape2 = grid2.shape
        # print(gridShape2)
        grid1NumRows = gridShape1[0]
        grid1NumCols = gridShape1[1]
        grid2NumRows = gridShape2[0]
        grid2NumCols = gridShape2[1]
        """backgroundColor1 = self.determineBackgroundColor(grid1)
        backgroundColor2 = self.determineBackgroundColor(grid2)"""
        backgroundColor1 = 0
        backgroundColor2 = 0 # not always true but not sure how to fix this and it's true in most csaes
        result = np.zeros((grid1NumRows, grid1NumCols))
        i = 0
        j = 0
        """print("non overlapping")
        print(grid1)
        print(grid2)"""
        while i < grid1NumRows:
            j = 0
            while j < grid1NumCols:
                grid1Curr = grid1[i, j]
                grid2Curr = grid2[i, j]
                if grid1Curr != backgroundColor1 or grid2Curr != backgroundColor2:
                    result[i, j] = backgroundColor1
                elif grid1Curr == backgroundColor1 and grid2Curr == backgroundColor2:
                    result[i, j] = outputColor
                j += 1
                # print("markoverlapping j ")
            i += 1
            # print("markoverlapping i ")
        return result

    def findShapes(self, grid):
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        i = 0
        shapesList = []
        adjacentCellsSet = set()
        adjacentCellsVisited = set()
        backgroundColor = self.determineBackgroundColor(grid)
        while i < numRows:
            j = 0
            while j < numCols:
                cellCurr = grid[i, j]
                cellCurrx = i
                cellCurry = j
                cellCoords = (cellCurrx, cellCurry)
                tupleCheck = (cellCurrx, cellCurry)
                if tupleCheck in adjacentCellsVisited:
                    j += 1
                    continue
                if cellCurr == backgroundColor:
                    adjacentCellsVisited.add(cellCoords)
                    j += 1
                    continue
                adjCells = self.findAdjacentCells(cellCoords, grid)
                k = 0
                adjCellsLen = len(adjCells)
                shapeCurr = set()
                while k < adjCellsLen:
                    cellCoordsAdded = adjCells[k]
                    cellx = cellCoordsAdded[0]
                    celly = cellCoordsAdded[1]
                    tupleCurr = (cellx, celly)
                    shapeCurr.add(tupleCurr)
                    adjacentCellsVisited.add(cellCoordsAdded)
                    k += 1
                shapesList.append(shapeCurr)
                j += 1
            i += 1
        return shapesList

    def sameShapeChangedColor(self, grid):
        gridShape = grid.shape
        numRows = gridShape[0]

    def allBackgroundColorToSpiral(self, grid, backgroundColorInput, mainColorOutput):
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        top = 0
        bottom = numRows - 1
        left = 0
        right = numCols - 1
        direction = "right"
        startingCell = grid[0, 0]
        outputGrid = deepcopy(grid)
        row = 0
        col = 0
        outputGrid[0, 0] = mainColorOutput
        gap = 1
        gapPlusOne = 1
        while bottom >= top and right >= left:
            while direction == "right":
                col += 1
                if col > right:
                    top += gapPlusOne
                    col -= 1
                    row += 1
                    direction = "down"
                    break
                outputGrid[row, col] = mainColorOutput
            while direction == "down":
                row += 1
                if row > bottom:
                    right -= gapPlusOne
                    row -= 1
                    col -= 1
                    direction = "left"
                    break
                outputGrid[row, col] = mainColorOutput
            while direction == "left":
                col -= 1
                if col < left:
                    col += 1
                    row -= 1
                    bottom -= gapPlusOne
                    direction = "up"
                    break
                outputGrid[row, col] = mainColorOutput
            while direction == "up":
                row -= 1
                if row < top:
                    row += 1
                    col += 1
                    left += gapPlusOne
                    direction = "right"
                    break
                outputGrid[row, col] = mainColorOutput
            # print(top, right, bottom, left)
        return outputGrid


    def numCellsSameColor(self, grid, color):
        gridShape = grid.shape
        numRows = gridShape[0]
        numCols = gridShape[1]
        i = 0
        result = 0
        while i < numRows:
            j = 0
            while j < numCols:
                cellCurr = grid[i, j]
                if cellCurr == color:
                    result += 1
                j += 1
            i += 1
        return result

    def rotate90DegCounterclockwise(self, input):
        output = np.rot90(input)
        return output

    def rotate180DegCounterclockwise(self, input):
        output90 = np.rot90(input)
        output = np.rot90(output90)
        return output

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

        testData = arc_problem.test_set().get_input_data().data()
        numTrainingDataSets = arc_problem.number_of_training_data_sets()
        trainingInputData = arc_problem.training_set()[0].get_input_data().data()
        predictions = [] # max of 3
        bestFitDict = dict()
        # 0520fde7
        i = 0
        maxPredictions = 3
        rotate90DegCounterclockwise = "Rotate90DegCounterclockwise"
        rotate180DegCounterclockwise = "Rotate180DegCounterclockwise"
        dividedVerticallyOverlappingShapes = "DividedVerticallyOverlappingShapes"
        dividedHorizontallyNonOverlappingShapes = "DividedHorizontallyNonOverlappingShapes"
        while i < numTrainingDataSets:
            trainingInputData = arc_problem.training_set()[i].get_input_data().data()
            shapesListInput = self.findShapes(trainingInputData)
            numCellsInput = len(shapesListInput)
            trainingOutputData = arc_problem.training_set()[i].get_output_data().data()
            shapesListOutput = self.findShapes(trainingOutputData)
            trainingInputShape = trainingInputData.shape
            trainingInputNumRows = trainingInputShape[0]
            trainingInputNumCols = trainingInputShape[1]
            trainingInputNumCells = trainingInputNumRows * trainingInputNumCols
            backgroundColorInput = self.determineBackgroundColor(trainingInputData)
            backgroundColorOutput = self.determineBackgroundColor(trainingOutputData)
            colorFirstCell = trainingInputData[0][0]
            backgroundColorNumCells = self.numCellsSameColor(trainingInputData, backgroundColorInput)
            """if colorFirstCell == backgroundColorInput and backgroundColorNumCells == trainingInputNumCells:
                bestFitDictVal = bestFitDict.get("BlankGridToSpiral", 0)
                bestFitDict["BlankGridToSpiral"] = bestFitDictVal + 1"""
            # i have to make this flipping and rotating stuf fmore generalizable so it's for specific parts of the grid not the entire grid
            # and also so i dont have separate functions for each degree just one which rotates a certain degree whatever at a time
            rot90Input = np.rot90(trainingInputData)
            rot90Output = np.rot90(trainingOutputData)
            trainingOutputNumRows = trainingOutputData.shape[0]
            trainingOutputNumCols = trainingOutputData.shape[1]
            if trainingInputNumRows == trainingOutputNumRows and trainingInputNumCols == trainingOutputNumCols:
                if np.array_equal(rot90Input, trainingOutputData):
                    # print(arc_problem._id)
                    bestFitDictVal = bestFitDict.get(rotate90DegCounterclockwise, 0)
                    bestFitDict[rotate90DegCounterclockwise] = bestFitDictVal + 1
                rot180Input = np.rot90(rot90Input)
                rot180Output = np.rot90(rot90Output)
                if np.array_equal(rot180Input, trainingOutputData):
                    bestFitDictVal = bestFitDict.get(rotate180DegCounterclockwise, 0)
                    bestFitDict[rotate180DegCounterclockwise] = bestFitDictVal + 1
            # print("num cols" + str(trainingInputNumCols) + arc_problem._id)
            if trainingInputNumRows % 2 == 1:
                # print("% 2 " + arc_problem._id)
                inputMiddleCols = trainingInputNumCols // 2
                # inputMiddleMinusOneCols = inputMiddleCols - 1
                inputMiddleRows = trainingInputNumRows // 2
                # inputMiddleMinusOneRows = inputMiddleRows - 1
                inputFirstHalf = trainingInputData[:, 0:inputMiddleCols]
                inputLastHalf = trainingInputData[:, inputMiddleCols + 1:]
                inputMiddleCol = trainingInputData[:, inputMiddleCols:]
                dividedVerticallyRow = 0
                dividedVerticallyCol = 0
                # i should move these to one helper fucntion later
                dividedVertically = True
                while dividedVerticallyRow < trainingInputNumRows:
                    cellColorDividedVertically = trainingInputData[dividedVerticallyRow, inputMiddleCols]
                    if cellColorDividedVertically == backgroundColorInput:
                        dividedVerticallyRow += 1
                        # print("background color " + arc_problem._id)
                        dividedVertically = False
                        break
                    if dividedVerticallyRow == 0:
                        cellColorDividedVerticallyPrev = cellColorDividedVertically
                        dividedVerticallyRow += 1
                        continue
                    if cellColorDividedVertically != cellColorDividedVerticallyPrev:
                        # print("not the same color " + arc_problem._id)
                        dividedVertically = False
                        break
                    cellColorDividedVerticallyPrev = cellColorDividedVertically
                    dividedVerticallyRow += 1
                if dividedVertically == True:
                    bestFitDictVal = bestFitDict.get(dividedVerticallyOverlappingShapes, 0)
                    bestFitDict[dividedVerticallyOverlappingShapes] = bestFitDictVal + 1
                    # print("divided vertically true" + arc_problem._id)

                # make this into a helper func and make it generalizable so i can do it for either vertically or horizontally tho this one's a bit different since
                # it's looking for non overlapping cells or whatever the math term is cells that are not filled in in both and the ones that are not filled in
                # turn green 6430c8c4.json
                # print("% 2 horizontal" + arc_problem._id)
                inputMiddleCols = trainingInputNumCols // 2
                # inputMiddleMinusOneCols = inputMiddleCols - 1
                inputMiddleRows = trainingInputNumRows // 2
                # inputMiddleMinusOneRows = inputMiddleRows - 1
                inputFirstHalf = trainingInputData[0:inputMiddleRows, :]
                inputLastHalf = trainingInputData[inputMiddleRows + 1:, :]
                # inputMiddleRow = trainingInputData[]
                dividedHorizontallyRow = 0
                dividedHorizontallyCol = 0
                # i should move these to one helper fucntion later
                dividedHorizontally = True
                backgroundColorInput = 0
                """if arc_problem._id == "6430c8c4":
                    print("1st loop before")
                    print(inputFirstHalf)
                    print("last half")
                    print(inputLastHalf)"""
                while dividedHorizontallyCol < trainingInputNumCols:
                    """if arc_problem._id == "6430c8c4":
                        print("1st loop")
                        print(inputFirstHalf)
                        print("last half")
                        print(inputLastHalf)"""
                    cellColorDividedHorizontally = trainingInputData[inputMiddleRows, dividedHorizontallyCol]
                    if cellColorDividedHorizontally == backgroundColorInput:
                        dividedHorizontallyRow += 1
                        # print("background color " + arc_problem._id)
                        dividedHorizontally = False
                        break
                    if dividedHorizontallyRow == 0:
                        cellColorDividedHorizontallyPrev = cellColorDividedHorizontally
                        dividedHorizontallyRow += 1
                        continue
                    if cellColorDividedHorizontally != cellColorDividedHorizontallyPrev:
                        # print("not the same color horizontal " + arc_problem._id)
                        dividedHorizontally = False
                        break
                    cellColorDividedHorizontallyPrev = cellColorDividedHorizontally
                    dividedHorizontallyCol += 1
                if dividedHorizontally == True:
                    bestFitDictVal = bestFitDict.get(dividedHorizontallyNonOverlappingShapes, 0)
                    bestFitDict[dividedHorizontallyNonOverlappingShapes] = bestFitDictVal + 1
                    # print("divided horizontally true" + arc_problem._id)
            # f76d97a5
            # if
            i += 1
        keys = bestFitDict.keys()
        bestFitList = []
        for key in keys:
            count = bestFitDict[key]
            tupleBestFit = (count, key)
            bestFitList.append(tupleBestFit)
        bestFitList.sort()
        j = 0
        correctTrainingOutputDict = dict()
        bestFitListLen = len(bestFitList)
        minMaxPredictionsBestFitLen = min(maxPredictions, bestFitListLen)
        while j < minMaxPredictionsBestFitLen:
            bestFitCurr = bestFitList[j]
            bestFitCurrCount = bestFitCurr[0]
            bestFitCurrKey = bestFitCurr[1]
            k = 0
            while k < numTrainingDataSets:
                currRes = np.array([]) #placeholder obviously
                trainingInputDataCheck = arc_problem.training_set()[k].get_input_data().data()
                trainingOutputDataCheck = arc_problem.training_set()[k].get_output_data().data()
                trainingInputDataShape = trainingInputDataCheck.shape
                trainingInputNumRows = trainingInputDataShape[0]
                trainingInputNumCols = trainingInputDataShape[1]
                trainingInputMiddleCols = trainingInputNumCols // 2
                trainingInputMiddleRows = trainingInputNumRows // 2
                if bestFitCurrKey == "BlankGridToSpiral":
                    outputColor = (arc_problem.training_set()[k].get_output_data().data())[0][0]
                    currRes = self.allBackgroundColorToSpiral(trainingInputDataCheck, backgroundColorInput, outputColor)
                    """if currRes == trainingOutputDataCheck:
                        currDictVal = correctTrainingOutputDict.get(bestFitCurrKey, 0)
                        correctTrainingOutputDict[bestFitCurrKey] = currDictVal + 1"""
                    blankGridToSpiralOutputColor = outputColor
                if bestFitCurrKey == rotate90DegCounterclockwise:
                    currRes = self.rotate90DegCounterclockwise(trainingInputDataCheck)
                    """if currRes == trainingOutputDataCheck:
                        currDictVal = correctTrainingOutputDict.get(bestFitCurrKey, 0)
                        correctTrainingOutputDict[bestFitCurrKey] = currDictVal + 1"""
                if bestFitCurrKey == rotate180DegCounterclockwise:
                    currRes = self.rotate180DegCounterclockwise(trainingInputDataCheck)
                if bestFitCurrKey == dividedVerticallyOverlappingShapes:
                    inputFirstHalf = trainingInputDataCheck[:, 0:trainingInputMiddleCols]
                    inputLastHalf = trainingInputDataCheck[:, trainingInputMiddleCols + 1:]
                    currResTuple = self.markOverlappingCellsTraining(inputFirstHalf, inputLastHalf, trainingOutputDataCheck)
                    currRes = currResTuple[0]
                    dividedVerticallyOverlappingShapesOutputColor = currResTuple[1]
                    """if arc_problem._id == "0520fde7":
                        print(currRes)
                        print(trainingOutputDataCheck)"""
                if bestFitCurrKey == dividedHorizontallyNonOverlappingShapes:
                    inputFirstHalf = trainingInputDataCheck[0:trainingInputMiddleRows, :]
                    inputLastHalf = trainingInputDataCheck[trainingInputMiddleRows + 1:, :]
                    inputFirstHalfShape = inputFirstHalf.shape
                    inputFirstHalfNumRows = inputFirstHalfShape[0]
                    inputLastHalfShape = inputLastHalf.shape
                    inputLastHalfNumRows = inputLastHalfShape[0]
                    """if arc_problem._id == "6430c8c4":
                        print(inputFirstHalf)
                        print("last half")
                        print(inputLastHalf)"""
                    if inputFirstHalfNumRows == inputLastHalfNumRows:
                        currResTuple = self.markNonOverlappingCellsHorizontallyTraining(inputFirstHalf, inputLastHalf, trainingOutputDataCheck)
                        currRes = currResTuple[0]
                        dividedHorizontallyNonOverlappingShapesOutputColor = currResTuple[1]
                        """if arc_problem._id == "0520fde7":
                            print(currRes)
                            print(trainingOutputDataCheck)"""
                if np.array_equal(currRes, trainingOutputDataCheck):
                    currDictVal = correctTrainingOutputDict.get(bestFitCurrKey, 0)
                    correctTrainingOutputDict[bestFitCurrKey] = currDictVal + 1
                k += 1
            j += 1
        keysCorrectDict = correctTrainingOutputDict.keys()
        correctList = []
        for keyCorrectDict in keysCorrectDict:
            numCorrect = correctTrainingOutputDict[keyCorrectDict]
            tupleCorrect = (numCorrect, keyCorrectDict)
            correctList.append(tupleCorrect)
        correctList.sort()
        b = 0
        correctListLen = len(correctList)
        minMaxPredictionsCorrectLen = min(maxPredictions, correctListLen)
        while b < minMaxPredictionsCorrectLen:
            correctCurrKey = correctList[b]
            correctCurrCount = correctCurrKey[0]
            correctCurrKeyFinal = correctCurrKey[1]
            backgroundColorTest = self.determineBackgroundColor(testData)
            testShape = testData.shape
            testNumRows = testShape[0]
            testNumCols = testShape[1]
            testMiddleCols = testNumCols // 2
            testMiddleRows = testNumRows // 2
            if correctCurrKeyFinal == "BlankGridToSpiral":
                currFinal = self.allBackgroundColorToSpiral(testData, backgroundColorInput, blankGridToSpiralOutputColor)
                predictions.append(currFinal)
            if correctCurrKeyFinal == rotate90DegCounterclockwise:
                currFinal = self.rotate90DegCounterclockwise(testData)
                predictions.append(currFinal)
            if correctCurrKeyFinal == rotate180DegCounterclockwise:
                currFinal = self.rotate180DegCounterclockwise(testData)
                predictions.append(currFinal)
            if correctCurrKeyFinal == dividedVerticallyOverlappingShapes:
                testFirstHalf = testData[:, 0:testMiddleCols]
                testLastHalf = testData[:, testMiddleCols + 1:]
                # print("color " + str(dividedVerticallyOverlappingShapesOutputColor))
                currFinal = self.markOverlappingCells(testFirstHalf, testLastHalf, dividedVerticallyOverlappingShapesOutputColor)
                predictions.append(currFinal)
            if correctCurrKeyFinal == dividedHorizontallyNonOverlappingShapes:
                testFirstHalf = testData[0:testMiddleRows, :]
                testLastHalf = testData[testMiddleRows + 1:, :]
                # print("color " + str(dividedHorizontallyNonOverlappingShapesOutputColor))
                """print("correct horizontally")
                print(testFirstHalf)
                print(testLastHalf)"""
                currFinal = self.markNonOverlappingCellsHorizontally(testFirstHalf, testLastHalf, dividedHorizontallyNonOverlappingShapesOutputColor)
                predictions.append(currFinal)
            b += 1
        return predictions


        # color invert one b194
        self.invertColors(grid)
        # c

        # mirror one f25

        # ce22 turn one cell into a larger square and change input color to output color

        # ed36, rotate 90 degrees

        # 615, rotate -180 degrees

        # 434, make everything in between black only leave the border use numpy indexing i guess if theres only like 1 or 2 cells

        # 623, fill diagonals
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
