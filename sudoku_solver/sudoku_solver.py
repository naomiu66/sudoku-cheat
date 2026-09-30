class SudokuSolver: 
    def __init__(self):
        self.n = 9
        self.b_n = 3

        self.rows = {i: 0 for i in range(self.n)}
        self.cols = {i: 0 for i in range(self.n)}
        self.blocks = {(i, j): 0 for i in range(self.b_n) for j in range(self.b_n)}

        self.empties = []
    
    def place(self, r, c, num):
        self.rows[r] |= num
        self.cols[c] |= num
        self.blocks[(r // 3, c // 3)] |= num
    
        
    def remove(self, r, c, num):
        self.rows[r] &= ~num
        self.cols[c] &= ~num
        self.blocks[(r // 3, c // 3)] &= ~num
        
    def bit_position(self, mask):
        pos = 0
        while (1 << pos) != mask:
            pos += 1
        return pos
       
    def backtrack(self, board: list[list[str]]):
        if not self.empties:
            return True

        min_options = 10
        idx = -1
        best_mask = 0

        for k, (r, c) in enumerate(self.empties):
            b = (r // 3, c // 3)
            used = self.rows[r] | self.cols[c] | self.blocks[b]
            candidates = (~used) & 0x1FF
            options = candidates.bit_count()

            if options < min_options:
                min_options = options
                idx = k
                best_mask = candidates
                if options == 1:
                    break

        r, c = self.empties.pop(idx)
        b = (r // 3, c // 3)

        cand = best_mask

        while cand:
            pick = cand & -cand
            num = self.bit_position(pick)

            self.place(r, c, pick)
            board[r][c] = str(num + 1)

            if self.backtrack():
                return True

            self.remove(r, c, pick)
            board[r][c] = "."

            cand -= pick 

        self.empties.insert(idx, (r, c))
        return False   
    
    def solveSudoku(self, board: list[list[str]]) -> list[list[str]]:
        self.__init__()

        for r in range(self.n):
            for c in range(self.n):
                val = board[r][c]
                if val != ".":
                    mask = 1 << (int(val) - 1)
                    self.place(r, c, mask)
                else:
                    self.empties.append((r, c))

        self.backtrack()
        
        return board