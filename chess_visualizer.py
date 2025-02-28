import pygame
import os
import numpy as np

class ChessVisualizer:
    def __init__(self, board_size=512):
        """
        Initialize the chess visualizer.
        
        Args:
            board_size: Size of the chess board in pixels
        """
        pygame.init()
        self.board_size = board_size
        self.square_size = board_size // 8
        self.screen = pygame.display.set_mode((board_size, board_size))
        pygame.display.set_caption("Chess Game Visualizer")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.LIGHT_SQUARE = (240, 217, 181)  # Light brown
        self.DARK_SQUARE = (181, 136, 99)    # Dark brown
        self.HIGHLIGHT = (255, 255, 0, 50)   # Yellow highlight with transparency
        
        # Load piece images
        self.pieces = {}
        self.load_pieces()
        
        # Game state
        self.last_move = None
        self.running = False
        self.clock = pygame.time.Clock()
        
    def load_pieces(self):
        """Load chess piece images"""
        piece_mapping = {
            1: 'wp',  # white pawn
            2: 'wn',  # white knight
            3: 'wb',  # white bishop
            4: 'wr',  # white rook
            5: 'wq',  # white queen
            6: 'wk',  # white king
            -1: 'bp', # black pawn
            -2: 'bn', # black knight
            -3: 'bb', # black bishop
            -4: 'br', # black rook
            -5: 'bq', # black queen
            -6: 'bk'  # black king
        }
        
        # Create a pieces directory if it doesn't exist
        if not os.path.exists('pieces'):
            os.makedirs('pieces')
            print("Created 'pieces' directory. Please add chess piece images to this directory.")
            print("Piece filenames should be: wp.png, wn.png, wb.png, wr.png, wq.png, wk.png, bp.png, bn.png, bb.png, br.png, bq.png, bk.png")
            
        # Try to load pieces, use colored rectangles as fallback
        for piece_value, piece_code in piece_mapping.items():
            piece_path = os.path.join('pieces', f'{piece_code}.png')
            try:
                if os.path.exists(piece_path):
                    img = pygame.image.load(piece_path)
                    self.pieces[piece_value] = pygame.transform.scale(img, (self.square_size, self.square_size))
                else:
                    # Create a colored rectangle as a fallback
                    color = self.WHITE if piece_value > 0 else self.BLACK
                    piece_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                    pygame.draw.rect(piece_surface, (*color, 180), (0, 0, self.square_size, self.square_size))
                    
                    # Add a letter to identify the piece
                    font = pygame.font.SysFont('Arial', self.square_size // 2)
                    piece_letter = piece_code[1].upper()
                    text = font.render(piece_letter, True, self.WHITE if piece_value < 0 else self.BLACK)
                    text_rect = text.get_rect(center=(self.square_size // 2, self.square_size // 2))
                    piece_surface.blit(text, text_rect)
                    
                    self.pieces[piece_value] = piece_surface
            except Exception as e:
                print(f"Error loading piece image {piece_code}: {e}")
                # Create a colored rectangle as a fallback
                color = self.WHITE if piece_value > 0 else self.BLACK
                piece_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                pygame.draw.rect(piece_surface, (*color, 180), (0, 0, self.square_size, self.square_size))
                
                # Add a letter to identify the piece
                font = pygame.font.SysFont('Arial', self.square_size // 2)
                piece_letter = piece_code[1].upper()
                text = font.render(piece_letter, True, self.WHITE if piece_value < 0 else self.BLACK)
                text_rect = text.get_rect(center=(self.square_size // 2, self.square_size // 2))
                piece_surface.blit(text, text_rect)
                
                self.pieces[piece_value] = piece_surface
    
    def draw_board(self):
        """Draw the chess board"""
        for row in range(8):
            for col in range(8):
                color = self.LIGHT_SQUARE if (row + col) % 2 == 0 else self.DARK_SQUARE
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    (col * self.square_size, row * self.square_size, self.square_size, self.square_size)
                )
                
        # Draw coordinates
        font = pygame.font.SysFont('Arial', self.square_size // 5)
        for i in range(8):
            # Draw row numbers (8 to 1)
            text = font.render(str(8 - i), True, self.LIGHT_SQUARE if i % 2 == 0 else self.DARK_SQUARE)
            self.screen.blit(text, (5, i * self.square_size + 5))
            
            # Draw column letters (a to h)
            text = font.render(chr(97 + i), True, self.LIGHT_SQUARE if (7 + i) % 2 == 0 else self.DARK_SQUARE)
            self.screen.blit(text, (i * self.square_size + self.square_size - 15, self.board_size - 15))
    
    def draw_pieces(self, board):
        """Draw the chess pieces on the board"""
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != 0:  # If there's a piece on this square
                    self.screen.blit(self.pieces[piece], (col * self.square_size, row * self.square_size))
    
    def highlight_last_move(self):
        """Highlight the last move made"""
        if self.last_move:
            start, end = self.last_move
            
            # Create a transparent surface for highlighting
            highlight_surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, (255, 255, 0, 100), highlight_surface.get_rect())
            
            # Highlight the start and end squares
            start_col, start_row = start[1], start[0]  # Convert from (row, col) to (col, row) for drawing
            end_col, end_row = end[1], end[0]
            
            self.screen.blit(highlight_surface, (start_col * self.square_size, start_row * self.square_size))
            self.screen.blit(highlight_surface, (end_col * self.square_size, end_row * self.square_size))
    
    def update(self, board, last_move=None):
        """Update the visualizer with the current board state"""
        if last_move:
            self.last_move = last_move
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        
        self.draw_board()
        self.highlight_last_move()
        self.draw_pieces(board)
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS
        return True
    
    def close(self):
        """Close the visualizer"""
        self.running = False
        pygame.quit() 