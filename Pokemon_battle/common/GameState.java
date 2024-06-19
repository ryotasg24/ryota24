package common;

public enum GameState {
    MY_TURN(3),
    OPPONENT_TURN(4),
    WIN(5),
    LOSE(6);

    private int id;

    public int getId() {
        return this.id;
    }

    private GameState(int id) {
        this.id = id;
    }
}
