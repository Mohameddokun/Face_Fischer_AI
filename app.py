import streamlit as st
import chess
import chess.svg
import torch
import torch.nn as nn
import numpy as np
import random
import base64
import os

# ==========================================
# 1. ARCHITECTURE DEFINITIONS (From Model)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = torch.relu(x + residual)
        return x

class ChessNet(nn.Module):
    def __init__(self, input_channels=19, num_blocks=4):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 73, kernel_size=1, bias=False), 
            nn.BatchNorm2d(73),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_blocks(x)
        return self.policy_head(x)

# ==========================================
# 2. TENSOR CONVERSIONS & INFERENCE ENGINE
# ==========================================
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    piece_map = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    tensor = torch.zeros(19, 8, 8, dtype=torch.float32)
    for sq, p in board.piece_map().items():
        row, col = divmod(sq, 8)
        ch = piece_map[p.piece_type] if p.color == chess.WHITE else 6+piece_map[p.piece_type]
        tensor[ch, row, col] = 1.0
    if board.turn == chess.WHITE: tensor[12,:,:] = 1.0
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        tensor[13, r, c] = 1.0
    cr = board.castling_rights
    tensor[14,:,:] = 1.0 if cr & chess.BB_H1 else 0.0
    tensor[15,:,:] = 1.0 if cr & chess.BB_A1 else 0.0
    tensor[16,:,:] = 1.0 if cr & chess.BB_H8 else 0.0
    tensor[17,:,:] = 1.0 if cr & chess.BB_A8 else 0.0
    tensor[18,:,:] = board.halfmove_clock / 100.0
    return tensor

def move_to_policy_index(move: chess.Move):
    fr, fc = divmod(move.from_square, 8)
    tr, tc = divmod(move.to_square, 8)
    dr, dc = tr - fr, tc - fc
    knight_offsets = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
    if (abs(dr), abs(dc)) in [(1,2), (2,1)]:
        try: return 56 + knight_offsets.index((dr, dc)), tr, tc
        except ValueError: return None
    dirs = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    for dir_idx, (ddr, ddc) in enumerate(dirs):
        if (ddr == 0 and ddc == 0) or (dr == 0 and dc == 0): continue
        if ddr != 0 and ddc != 0:
            if abs(dr) != abs(dc): continue
            if dr//abs(dr) != ddr or dc//abs(dc) != ddc: continue
            dist = abs(dr)
        elif ddr == 0:
            if dr != 0: continue
            if dc//abs(dc) != ddc: continue
            dist = abs(dc)
        else:
            if dc != 0: continue
            if dr//abs(dr) != ddr: continue
            dist = abs(dr)
        if 1 <= dist <= 7: return dir_idx * 7 + (dist - 1), tr, tc
    if move.promotion and move.promotion != chess.QUEEN:
        if tr == 7: forward_dr, left_dc, right_dc = -1, -1, 1
        elif tr == 0: forward_dr, left_dc, right_dc = 1, 1, -1
        else: return None
        if dr == forward_dr and dc == 0: dir_idx = 1
        elif dr == forward_dr and dc == left_dc: dir_idx = 0
        elif dr == forward_dr and dc == right_dc: dir_idx = 2
        else: return None
        piece_offset = 0 if move.promotion == chess.KNIGHT else (3 if move.promotion == chess.BISHOP else 6)
        return 64 + piece_offset + dir_idx, tr, tc
    return None

@st.cache_resource
def load_trained_model():
    model = ChessNet(input_channels=19, num_blocks=4)
    model_path = "chess_resnet.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        st.sidebar.success("♟️ Bobby Fischer engine loaded successfully!")
    else:
        st.sidebar.warning("⚠️ 'chess_resnet.pth' not found. Running with baseline initializations.")
    model.eval()
    return model

def pick_move(model, board, temperature=0.1):
    with torch.no_grad():
        tensor = board_to_tensor(board).unsqueeze(0)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None
    move_probs = []
    for move in legal_moves:
        idx = move_to_policy_index(move)
        move_probs.append(probs[idx[0]*64 + idx[1]*8 + idx[2]] if idx is not None else 0.0)
    move_probs = np.array(move_probs)
    if move_probs.sum() > 0: move_probs /= move_probs.sum()
    else: return random.choice(legal_moves)
    
    if temperature > 0:
        move_probs = np.exp(np.log(move_probs + 1e-9) / temperature)
        return np.random.choice(legal_moves, p=move_probs / move_probs.sum())
    return legal_moves[np.argmax(move_probs)]

# ==========================================
# 3. STREAMLIT INTERFACE & STATE MANAGEMENT
# ==========================================
st.set_page_config(page_title="Play vs Bobby Fischer", page_icon="♟️", layout="centered")

# Track interactive board clicks
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "player_color" not in st.session_state:
    st.session_state.player_color = "White"
if "selected_square" not in st.session_state:
    st.session_state.selected_square = None

model = load_trained_model()

st.title("♟️ Face Bobby Fischer AI")
st.write("Click squares on the control panel below the board to coordinate your moves.")

# Sidebar Controls
st.sidebar.title("Match Setup")
color_selection = st.sidebar.selectbox("Choose your pieces:", ["White", "Black"], index=0 if st.session_state.player_color == "White" else 1)

if color_selection != st.session_state.player_color:
    st.session_state.player_color = color_selection
    st.session_state.board = chess.Board()
    st.session_state.selected_square = None
    st.rerun()

if st.sidebar.button("🔄 Reset Match"):
    st.session_state.board = chess.Board()
    st.session_state.selected_square = None
    st.rerun()

board = st.session_state.board
is_game_over = board.is_game_over(claim_draw=True)
is_ai_turn = (board.turn == chess.WHITE and st.session_state.player_color == "Black") or \
             (board.turn == chess.BLACK and st.session_state.player_color == "White")

# Render Global Main Board
def render_svg_board(chess_board):
    flipped = (st.session_state.player_color == "Black")
    svg_data = chess.svg.board(board=chess_board, size=400, flipped=flipped)
    b64_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    st.markdown(
        f'<div style="display: flex; justify-content: center; margin-bottom: 20px;"><img src="data:image/svg+xml;base64,{b64_svg}" width="400"/></div>',
        unsafe_allow_html=True
    )

render_svg_board(board)

# Game Status Banner
if not is_game_over:
    if is_ai_turn:
        st.info("🤖 **Bobby Fischer is calculating...**")
        ai_move = pick_move(model, board, temperature=0.1)
        if ai_move:
            board.push(ai_move)
            st.rerun()
    else:
        # Check if a square is currently selected to guide the user
        if st.session_state.selected_square is not None:
            current_sq_name = chess.square_name(st.session_state.selected_square)
            st.success(f"🟢 Selected **{current_sq_name.upper()}**. Click destination square to move.")
        else:
            st.success("🟢 **Your Turn:** Click any piece to start your move.")

        # ==========================================
        # 4. CLICK-TO-MOVE BUTTON MATRIX GRAPHIC
        # ==========================================
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        
        # Piece Unicode mapping for visual button indicators
        unicode_pieces = {
            'R': 'Piece: ♖', 'N': 'Piece: ♘', 'B': 'Piece: ♗', 'Q': 'Piece: ♕', 'K': 'Piece: ♔', 'P': 'Piece: ♙',
            'r': 'Piece: ♜', 'n': 'Piece: ♞', 'b': 'Piece: ♝', 'q': 'Piece: ♛', 'k': 'Piece: ♚', 'p': 'Piece: ♟'
        }

        # Flip perspective rows/cols to perfectly match the standard viewer orientation
        rank_indices = range(7, -1, -1) if st.session_state.player_color == "White" else range(8)
        file_indices = range(8) if st.session_state.player_color == "White" else range(7, -1, -1)

        st.write("### 🎛️ Interactive Controller Matrix")
        
        for r in rank_indices:
            cols = st.columns(8)
            for i, c in enumerate(file_indices):
                sq_idx = r * 8 + c
                piece = board.piece_at(sq_idx)
                sq_name = files[c] + ranks[r]
                
                # Format string text inside block
                lbl = unicode_pieces[piece.symbol()].split(": ")[1] if piece else "·"
                button_text = f"**{lbl}**\n{sq_name.upper()}"
                
                # visually highlight active square via border adjustments
                if st.session_state.selected_square == sq_idx:
                    button_text = f"👉 {sq_name.upper()}"

                if cols[i].button(button_text, key=f"sq_{sq_idx}", use_container_width=True):
                    if st.session_state.selected_square is None:
                        # First Click: Select matching player color piece
                        player_is_white = (st.session_state.player_color == "White")
                        if piece and piece.color == (chess.WHITE if player_is_white else chess.BLACK):
                            st.session_state.selected_square = sq_idx
                            st.rerun()
                    else:
                        # Second Click: Build intended coordinate conversion
                        from_sq = st.session_state.selected_square
                        to_sq = sq_idx
                        
                        # Build candidate moves (Auto-Promote Pawns to Queen to avoid input lock)
                        move = chess.Move(from_sq, to_sq)
                        if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.PAWN and r in [0, 7]:
                            move.promotion = chess.QUEEN
                        
                        if move in board.legal_moves:
                            board.push(move)
                            st.session_state.selected_square = None
                            st.rerun()
                        else:
                            # Re-click another piece of your own color to switch active selection seamlessly
                            player_is_white = (st.session_state.player_color == "White")
                            if piece and piece.color == (chess.WHITE if player_is_white else chess.BLACK):
                                st.session_state.selected_square = sq_idx
                            else:
                                st.session_state.selected_square = None
                            st.rerun()
else:
    st.error(f"🏁 **Game Over! Final Result: {board.result()}**")
    st.session_state.selected_square = None

# ==========================================
# 5. MOVELOG ENGINE (CRASH FIX IMPLEMENTED)
# ==========================================
if len(board.move_stack) > 0:
    st.write("---")
    with st.expander("📊 View Complete Game Log"):
        # FIX: Progressively track history via clean tracking instance to avoid string parsing crashes
        replay_board = chess.Board()
        safe_moves_san = []
        for historical_move in board.move_stack:
            safe_moves_san.append(replay_board.san(historical_move))
            replay_board.push(historical_move)
            
        formatted_history = []
        for i in range(0, len(safe_moves_san), 2):
            move_num = (i // 2) + 1
            w_move = safe_moves_san[i]
            b_move = safe_moves_san[i+1] if (i+1) < len(safe_moves_san) else ""
            formatted_history.append(f"**{move_num}.** {w_move} &nbsp;&nbsp;&nbsp;&nbsp; {b_move}")
        st.markdown("<br>".join(formatted_history), unsafe_allow_html=True)
