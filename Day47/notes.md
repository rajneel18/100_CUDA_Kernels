# Game of Life 

## Concept

The Game of Life is a cellular automaton devised by John Conway. It simulates the evolution of a population of cells based on simple rules.

## Rules

1.  **Survival:** A live cell with 2 or 3 live neighbors survives.
2.  **Death:** A live cell with fewer than 2 or more than 3 live neighbors dies.
3.  **Birth:** A dead cell with exactly 3 live neighbors becomes a live cell.

## CUDA Implementation

### Key Concepts

* **Parallel Processing:** Each cell's state can be updated independently, making it highly parallelizable.
* **Neighborhood Calculation:** Efficiently calculating the number of live neighbors is crucial.
* **Boundary Conditions:** Handling boundary cells requires special attention.
* **Double Buffering:** To avoid race conditions, use double buffering, where one buffer holds the current state, and the other holds the next state.
