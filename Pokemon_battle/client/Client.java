package client;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class Client {
    private static final int PORT = 8080;
    Socket socket = null;
    BufferedReader in = null; // サーバからの受信
    PrintWriter out = null; // サーバーに送信
    BufferedReader userIn = null; // ユーザーからの入力 コンソールなら標準入力

    private static final int MY_TURN = 3;
    private static final int OPPONENT_TURN = 4;
    private static final int WIN = 5;
    private static final int LOSE = 6;
    private int state;


    // クライアント生成時にユーザ入力を定義する。
    Client(BufferedReader userIn) {
        this.userIn = userIn;
        try{
            socket = new Socket("localhost", PORT);
            in = new BufferedReader(
                    new InputStreamReader(socket.getInputStream()));
            out = new PrintWriter(socket.getOutputStream(), true);
        }catch(IOException e){
            logging("接続エラー");
        }
    }

    /*
     * 送受信、ログ、　入力関連の設定
     */
    public void logging(Object s) { 
        System.out.println(s);
    }

    public String read() throws IOException { // 定義したユーザ入力から入力を読む
        return userIn.readLine();
    }

    public String receive() throws IOException { 
        return in.readLine();
    }

    // 行数(line)分受け取ってユーザ出力に出力
    public void receiveAndLog(int line) throws IOException { 
        for (int i = 0; i < line; i++) {
            logging(receive());
        }
    }

    public void send(String s) {
        out.println(s);
    }

    /*
     * ユーザ入力の判定関連
     */
    private boolean isValidYesNoInput(String val) {
        return val.equals("Yes") ||
                val.equals("No") ||
                val.equals("yes") ||
                val.equals("no");
    }

    private boolean isValidMoveInput(String str) {
        if (str.equals("1") ||
                str.equals("2") ||
                str.equals("3") ||
                str.equals("4")) {
            return true;
        }
        return false;
    }

    private boolean isValidTypeInput(String str) {
        if (str.equals("1") ||
                str.equals("2") ||
                str.equals("3") ||
                str.equals("4") ||
                str.equals("5")) {
            return true;
        }
        return false;
    }

    private boolean isValidStatsInput(String str) {
        if (str.equals("1") ||
                str.equals("2") ||
                str.equals("3") ||
                str.equals("4") ||
                str.equals("5") ||
                str.equals("6")) {
            return true;
        }
        return false;
    }

    /*
    * ユーザ入力関連
    */
    public boolean yesNoInput() throws IOException {
        String s = read();
        while (!isValidYesNoInput(s)) {
            logging("Yes or Noを入力してください");
            s = read();
        }
        if (s.equals("Yes") || s.equals("yes")) {
            return true;
        } else {
            return false;
        }
    }

    // 5種類のinputType(入力の種類) : move, type, name, password, stats
    public String input(String InputType) throws IOException {
        while (userIn.ready()) { // 相手のターン中の入力を捨てる。StackOverFlow
            read();
        }
        if(InputType.equals("move")){
            String s = read();
            while (!isValidMoveInput(s)) {
                logging("1~4の数字を入力してください");
                s = read();
            }
            return s;
        }
        if(InputType.equals("type")){
            String s = read();
            while (!isValidTypeInput(s)) {
                logging("1: 火, 2: 水, 3: 草, 4: 光, 5: 闇");
                logging("1~5の数字を入力してください");
                s = read();
            }
            return s;
        }
        if(InputType.equals("status")){
            String s = read();
            while (!isValidStatsInput(s)) {
                logging("1: hp, 2: 攻撃, 3: 防御, 4: 特攻, 5: 特防 6:素早さ");
                logging("1~6の数字を入力してください");
                s = read();
            }
            return s;
        }

        String tmp = "";
        if (InputType.equals("name")) {
            tmp = "名前";
        } else if (InputType.equals("password")) {
            tmp = "パスワード";
        }
        String str = read();
        while (str == null || str.equals("")) {
            logging(tmp + "を1文字以上で入力してください");
            str = read();
        }
        return str;
    }
    
    public int getState() {
        return state;
    }

    public void setState(int state) {
        this.state = state;
    }
}