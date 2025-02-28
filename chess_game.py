import pygame
import sys
from chess_engine import ChessEngine

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 512, 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 0, 0)

def load_images():
    """Load the chess piece images"""
    pieces = ['wp', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bp', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        # Use Unicode chess symbols instead of images
        IMAGES[piece] = None

def draw_board(screen):
    """Draw the chess board"""
    colors = [WHITE, GRAY]
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    """Draw the chess pieces on the board"""
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != 0:
                font = pygame.font.SysFont('Arial', 36)
                piece_symbol = get_piece_symbol(piece)
                text = font.render(piece_symbol, True, BLACK if piece > 0 else BLACK)
                screen.blit(text, pygame.Rect(col * SQ_SIZE + SQ_SIZE // 4, row * SQ_SIZE + SQ_SIZE // 4, SQ_SIZE, SQ_SIZE))

def get_piece_symbol(piece):
    """Returns the Unicode symbol for a piece"""
    symbols = {
        0: '',
        1: '♙', -1: '♟',
        2: '♘', -2: '♞',
        3: '♗', -3: '♝',
        4: '♖', -4: '♜',
        5: '♕', -5: '♛',
        6: '♔', -6: '♚'
    }
    return symbols[piece]

def highlight_squares(screen, game_state, valid_moves, selected_square):
    """Highlight the selected square and valid moves"""
    if selected_square != ():
        row, col = selected_square
        # Highlight selected square
        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)  # transparency value
        s.fill(YELLOW)
        screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))
        
        # Highlight valid moves
        s.fill(BLUE)
        for move in valid_moves:
            if move[0] == selected_square:
                screen.blit(s, (move[1][1] * SQ_SIZE, move[1][0] * SQ_SIZE))

def draw_game_state(screen, game_state, valid_moves, selected_square):
    """Draw the complete game state"""
    draw_board(screen)
    highlight_squares(screen, game_state, valid_moves, selected_square)
    draw_pieces(screen, game_state['board'])

def draw_text(screen, text):
    """Draw text on the screen"""
    font = pygame.font.SysFont("Arial", 32, True, False)
    text_object = font.render(text, True, BLACK)
    text_location = pygame.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH // 2 - text_object.get_width() // 2, 
                                                         HEIGHT // 2 - text_object.get_height() // 2)
    screen.blit(text_object, text_location)

def main():
    """Main function to run the chess game"""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    screen.fill(WHITE)
    game_engine = ChessEngine()
    load_images()
    
    running = True
    selected_square = ()  # (row, col) of the selected square
    player_clicks = []  # keeps track of player clicks (two tuples: [(start_row, start_col), (end_row, end_col)])
    valid_moves = game_engine.get_valid_moves()
    move_made = False  # flag for when a move is made
    game_over = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Mouse handler
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not game_over:
                    location = pygame.mouse.get_pos()  # (x, y) location of the mouse
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    
                    # Check if the same square is selected twice
                    if selected_square == (row, col):
                        selected_square = ()  # deselect
                        player_clicks = []
                    else:
                        selected_square = (row, col)
                        player_clicks.append(selected_square)
                    
                    # After second click, make a move
                    if len(player_clicks) == 2:
                        move = (player_clicks[0], player_clicks[1])
                        
                        # Check if the move is valid
                        valid_move = False
                        for valid in valid_moves:
                            if valid[0] == move[0] and valid[1] == move[1]:
                                game_engine.make_move(move[0], move[1])
                                move_made = True
                                selected_square = ()
                                player_clicks = []
                                valid_move = True
                                break
                        
                        if not valid_move:
                            player_clicks = [selected_square]
            
            # Key handler
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # reset the game
                    game_engine.reset_board()
                    valid_moves = game_engine.get_valid_moves()
                    selected_square = ()
                    player_clicks = []
                    move_made = False
                    game_over = False
                
                if event.key == pygame.K_u and len(game_engine.move_log) > 0:  # undo move
                    game_engine.undo_move()
                    move_made = True
                    game_over = False
                
                if event.key == pygame.K_q:  # quit the game
                    running = False
        
        # Update the game state if a move was made
        if move_made:
            valid_moves = game_engine.get_valid_moves()
            move_made = False
        
        # Draw the game state
        game_state = game_engine.get_game_state()
        draw_game_state(screen, game_state, valid_moves, selected_square)
        
        # Check for checkmate or stalemate
        if game_state['checkmate']:
            game_over = True
            draw_text(screen, "Checkmate! " + ("Black" if game_state['white_to_move'] else "White") + " wins!")
        elif game_state['stalemate']:
            game_over = True
            draw_text(screen, "Stalemate!")
        
        clock.tick(MAX_FPS)
        pygame.display.flip()

if __name__ == "__main__":
    main() 