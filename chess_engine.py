import numpy as np

class ChessEngine:
    def __init__(self):
        # Initialize the board
        # 0 = empty, 1-6 = white pieces, -1 to -6 = black pieces
        # 1/-1 = pawn, 2/-2 = knight, 3/-3 = bishop, 4/-4 = rook, 5/-5 = queen, 6/-6 = king
        self.board = np.zeros((8, 8), dtype=int)
        self.reset_board()
        
        # Game state
        self.white_to_move = True
        self.move_log = []
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.checkmate = False
        self.stalemate = False
        
        # Castling rights
        self.white_king_side_castle = True
        self.white_queen_side_castle = True
        self.black_king_side_castle = True
        self.black_queen_side_castle = True
        
        # En passant
        self.en_passant_possible = ()  # coordinates where en passant capture is possible
        
    def reset_board(self):
        # Set up the initial board position
        # Black pieces
        self.board[0] = np.array([-4, -2, -3, -5, -6, -3, -2, -4])
        self.board[1] = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
        
        # White pieces
        self.board[6] = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.board[7] = np.array([4, 2, 3, 5, 6, 3, 2, 4])
        
        # Reset game state
        self.white_to_move = True
        self.move_log = []
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.checkmate = False
        self.stalemate = False
        self.white_king_side_castle = True
        self.white_queen_side_castle = True
        self.black_king_side_castle = True
        self.black_queen_side_castle = True
        self.en_passant_possible = ()
    
    def make_move(self, start, end):
        """
        Makes a move on the board
        start: tuple (row, col) of starting position
        end: tuple (row, col) of ending position
        """
        start_row, start_col = start
        end_row, end_col = end
        
        # Store move in log
        self.move_log.append((start, end, self.board[start_row][start_col], self.board[end_row][end_col]))
        
        # Update king position if king is moved
        if self.board[start_row][start_col] == 6:  # white king
            self.white_king_location = (end_row, end_col)
            self.white_king_side_castle = False
            self.white_queen_side_castle = False
        elif self.board[start_row][start_col] == -6:  # black king
            self.black_king_location = (end_row, end_col)
            self.black_king_side_castle = False
            self.black_queen_side_castle = False
        
        # Update castling rights if rook is moved
        if start_row == 7 and start_col == 0:  # white queen-side rook
            self.white_queen_side_castle = False
        elif start_row == 7 and start_col == 7:  # white king-side rook
            self.white_king_side_castle = False
        elif start_row == 0 and start_col == 0:  # black queen-side rook
            self.black_queen_side_castle = False
        elif start_row == 0 and start_col == 7:  # black king-side rook
            self.black_king_side_castle = False
        
        # Handle castling move
        if abs(self.board[start_row][start_col]) == 6 and abs(start_col - end_col) == 2:
            if end_col - start_col == 2:  # king-side castle
                # Move the rook
                self.board[end_row][end_col-1] = self.board[end_row][end_col+1]
                self.board[end_row][end_col+1] = 0
            else:  # queen-side castle
                # Move the rook
                self.board[end_row][end_col+1] = self.board[end_row][end_col-2]
                self.board[end_row][end_col-2] = 0
        
        # Handle en passant capture
        if abs(self.board[start_row][start_col]) == 1 and (start_col != end_col) and self.board[end_row][end_col] == 0:
            # This is a diagonal pawn move to an empty square - must be en passant
            self.board[start_row][end_col] = 0  # Capture the pawn
        
        # Set en passant possibility
        if abs(self.board[start_row][start_col]) == 1 and abs(start_row - end_row) == 2:
            self.en_passant_possible = ((start_row + end_row) // 2, start_col)
        else:
            self.en_passant_possible = ()
        
        # Move the piece
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = 0
        
        # Pawn promotion (automatically to queen for simplicity)
        if self.board[end_row][end_col] == 1 and end_row == 0:  # white pawn reaches the end
            self.board[end_row][end_col] = 5
        elif self.board[end_row][end_col] == -1 and end_row == 7:  # black pawn reaches the end
            self.board[end_row][end_col] = -5
        
        # Switch turns
        self.white_to_move = not self.white_to_move
        
        # Check for checkmate or stalemate
        self.check_game_state()
        
    def undo_move(self):
        """Undoes the last move"""
        if len(self.move_log) == 0:
            return
        
        start, end, moved_piece, captured_piece = self.move_log.pop()
        start_row, start_col = start
        end_row, end_col = end
        
        # Restore the board state
        self.board[start_row][start_col] = moved_piece
        self.board[end_row][end_col] = captured_piece
        
        # Restore king position if king was moved
        if moved_piece == 6:  # white king
            self.white_king_location = (start_row, start_col)
        elif moved_piece == -6:  # black king
            self.black_king_location = (start_row, start_col)
        
        # Switch turns back
        self.white_to_move = not self.white_to_move
        
        # Reset checkmate and stalemate
        self.checkmate = False
        self.stalemate = False
    
    def get_valid_moves(self):
        """Returns all valid moves for the current player"""
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                # Check if the piece belongs to the current player
                if (self.white_to_move and piece > 0) or (not self.white_to_move and piece < 0):
                    moves.extend(self.get_piece_moves((row, col)))
        return moves
    
    def get_piece_moves(self, position):
        """Returns all valid moves for a piece at the given position"""
        row, col = position
        piece = self.board[row][col]
        moves = []
        
        # Pawn moves
        if abs(piece) == 1:
            moves.extend(self.get_pawn_moves(position))
        # Knight moves
        elif abs(piece) == 2:
            moves.extend(self.get_knight_moves(position))
        # Bishop moves
        elif abs(piece) == 3:
            moves.extend(self.get_bishop_moves(position))
        # Rook moves
        elif abs(piece) == 4:
            moves.extend(self.get_rook_moves(position))
        # Queen moves
        elif abs(piece) == 5:
            moves.extend(self.get_bishop_moves(position))
            moves.extend(self.get_rook_moves(position))
        # King moves
        elif abs(piece) == 6:
            moves.extend(self.get_king_moves(position))
        
        return moves
    
    def get_pawn_moves(self, position):
        """Returns all valid moves for a pawn at the given position"""
        row, col = position
        piece = self.board[row][col]
        moves = []
        
        if self.white_to_move:  # white pawn
            # Move forward one square
            if row > 0 and self.board[row-1][col] == 0:
                moves.append(((row, col), (row-1, col)))
                # Move forward two squares from starting position
                if row == 6 and self.board[row-2][col] == 0:
                    moves.append(((row, col), (row-2, col)))
            
            # Capture diagonally
            if row > 0 and col > 0 and self.board[row-1][col-1] < 0:
                moves.append(((row, col), (row-1, col-1)))
            if row > 0 and col < 7 and self.board[row-1][col+1] < 0:
                moves.append(((row, col), (row-1, col+1)))
            
            # En passant
            if self.en_passant_possible and row == 3:
                ep_row, ep_col = self.en_passant_possible
                if ep_col == col - 1 or ep_col == col + 1:
                    moves.append(((row, col), (ep_row, ep_col)))
        
        else:  # black pawn
            # Move forward one square
            if row < 7 and self.board[row+1][col] == 0:
                moves.append(((row, col), (row+1, col)))
                # Move forward two squares from starting position
                if row == 1 and self.board[row+2][col] == 0:
                    moves.append(((row, col), (row+2, col)))
            
            # Capture diagonally
            if row < 7 and col > 0 and self.board[row+1][col-1] > 0:
                moves.append(((row, col), (row+1, col-1)))
            if row < 7 and col < 7 and self.board[row+1][col+1] > 0:
                moves.append(((row, col), (row+1, col+1)))
            
            # En passant
            if self.en_passant_possible and row == 4:
                ep_row, ep_col = self.en_passant_possible
                if ep_col == col - 1 or ep_col == col + 1:
                    moves.append(((row, col), (ep_row, ep_col)))
        
        return moves
    
    def get_knight_moves(self, position):
        """Returns all valid moves for a knight at the given position"""
        row, col = position
        piece = self.board[row][col]
        moves = []
        
        # Knight move offsets
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for move in knight_moves:
            end_row = row + move[0]
            end_col = col + move[1]
            
            # Check if the move is on the board
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                
                # Check if the square is empty or contains an enemy piece
                if end_piece == 0 or (piece > 0 and end_piece < 0) or (piece < 0 and end_piece > 0):
                    moves.append(((row, col), (end_row, end_col)))
        
        return moves
    
    def get_bishop_moves(self, position):
        """Returns all valid moves for a bishop at the given position"""
        row, col = position
        piece = self.board[row][col]
        moves = []
        
        # Bishop move directions (diagonals)
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for direction in directions:
            for i in range(1, 8):
                end_row = row + direction[0] * i
                end_col = col + direction[1] * i
                
                # Check if the move is on the board
                if not (0 <= end_row < 8 and 0 <= end_col < 8):
                    break
                
                end_piece = self.board[end_row][end_col]
                
                # Check if the square is empty
                if end_piece == 0:
                    moves.append(((row, col), (end_row, end_col)))
                # Check if the square contains an enemy piece
                elif (piece > 0 and end_piece < 0) or (piece < 0 and end_piece > 0):
                    moves.append(((row, col), (end_row, end_col)))
                    break
                # Square contains a friendly piece
                else:
                    break
        
        return moves
    
    def get_rook_moves(self, position):
        """Returns all valid moves for a rook at the given position"""
        row, col = position
        piece = self.board[row][col]
        moves = []
        
        # Rook move directions (horizontal and vertical)
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        
        for direction in directions:
            for i in range(1, 8):
                end_row = row + direction[0] * i
                end_col = col + direction[1] * i
                
                # Check if the move is on the board
                if not (0 <= end_row < 8 and 0 <= end_col < 8):
                    break
                
                end_piece = self.board[end_row][end_col]
                
                # Check if the square is empty
                if end_piece == 0:
                    moves.append(((row, col), (end_row, end_col)))
                # Check if the square contains an enemy piece
                elif (piece > 0 and end_piece < 0) or (piece < 0 and end_piece > 0):
                    moves.append(((row, col), (end_row, end_col)))
                    break
                # Square contains a friendly piece
                else:
                    break
        
        return moves
    
    def get_king_moves(self, position):
        """Returns all valid moves for a king at the given position"""
        row, col = position
        piece = self.board[row][col]
        moves = []
        
        # King move directions (all 8 surrounding squares)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for direction in directions:
            end_row = row + direction[0]
            end_col = col + direction[1]
            
            # Check if the move is on the board
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                
                # Check if the square is empty or contains an enemy piece
                if end_piece == 0 or (piece > 0 and end_piece < 0) or (piece < 0 and end_piece > 0):
                    moves.append(((row, col), (end_row, end_col)))
        
        # Castling
        if self.white_to_move and piece == 6:
            # King-side castling
            if self.white_king_side_castle and self.board[7][5] == 0 and self.board[7][6] == 0:
                moves.append(((7, 4), (7, 6)))
            # Queen-side castling
            if self.white_queen_side_castle and self.board[7][1] == 0 and self.board[7][2] == 0 and self.board[7][3] == 0:
                moves.append(((7, 4), (7, 2)))
        elif not self.white_to_move and piece == -6:
            # King-side castling
            if self.black_king_side_castle and self.board[0][5] == 0 and self.board[0][6] == 0:
                moves.append(((0, 4), (0, 6)))
            # Queen-side castling
            if self.black_queen_side_castle and self.board[0][1] == 0 and self.board[0][2] == 0 and self.board[0][3] == 0:
                moves.append(((0, 4), (0, 2)))
        
        return moves
    
    def is_in_check(self, white):
        """Returns True if the given player is in check"""
        if white:
            king_row, king_col = self.white_king_location
        else:
            king_row, king_col = self.black_king_location
        
        # Temporarily switch turns to get opponent's moves
        self.white_to_move = not white
        opponent_moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                # Check if the piece belongs to the opponent
                if (white and piece < 0) or (not white and piece > 0):
                    opponent_moves.extend(self.get_piece_moves((row, col)))
        
        # Switch turns back
        self.white_to_move = white
        
        # Check if any opponent move can capture the king
        for move in opponent_moves:
            if move[1] == (king_row, king_col):
                return True
        
        return False
    
    def check_game_state(self):
        """Checks if the game is in checkmate or stalemate"""
        valid_moves = self.get_valid_moves()
        
        # Filter out moves that would put or leave the king in check
        legal_moves = []
        for move in valid_moves:
            # Make the move
            start, end = move
            start_row, start_col = start
            end_row, end_col = end
            
            # Save the current state
            temp_piece = self.board[end_row][end_col]
            self.board[end_row][end_col] = self.board[start_row][start_col]
            self.board[start_row][start_col] = 0
            
            # Update king position if king is moved
            king_moved = False
            if self.board[end_row][end_col] == 6:  # white king
                old_king_pos = self.white_king_location
                self.white_king_location = (end_row, end_col)
                king_moved = True
            elif self.board[end_row][end_col] == -6:  # black king
                old_king_pos = self.black_king_location
                self.black_king_location = (end_row, end_col)
                king_moved = True
            
            # Check if the king is in check after the move
            in_check = self.is_in_check(self.white_to_move)
            
            # Restore the board state
            self.board[start_row][start_col] = self.board[end_row][end_col]
            self.board[end_row][end_col] = temp_piece
            
            # Restore king position if king was moved
            if king_moved:
                if self.white_to_move:
                    self.white_king_location = old_king_pos
                else:
                    self.black_king_location = old_king_pos
            
            # If the move doesn't leave the king in check, it's legal
            if not in_check:
                legal_moves.append(move)
        
        # Check if the current player is in check
        in_check = self.is_in_check(self.white_to_move)
        
        # If no legal moves and in check, it's checkmate
        if len(legal_moves) == 0 and in_check:
            self.checkmate = True
        # If no legal moves and not in check, it's stalemate
        elif len(legal_moves) == 0:
            self.stalemate = True
        
    def get_piece_symbol(self, piece):
        """Returns the symbol for a piece"""
        symbols = {
            0: '.',
            1: '♙', -1: '♟',
            2: '♘', -2: '♞',
            3: '♗', -3: '♝',
            4: '♖', -4: '♜',
            5: '♕', -5: '♛',
            6: '♔', -6: '♚'
        }
        return symbols[piece]
    
    def print_board(self):
        """Prints the current board state"""
        print('  a b c d e f g h')
        print(' +-----------------+')
        for row in range(8):
            print(f'{8-row}|', end=' ')
            for col in range(8):
                print(self.get_piece_symbol(self.board[row][col]), end=' ')
            print(f'|{8-row}')
        print(' +-----------------+')
        print('  a b c d e f g h')
        
    def get_game_state(self):
        """Returns the current game state"""
        return {
            'board': self.board.copy(),
            'white_to_move': self.white_to_move,
            'checkmate': self.checkmate,
            'stalemate': self.stalemate,
            'white_king_location': self.white_king_location,
            'black_king_location': self.black_king_location
        } 