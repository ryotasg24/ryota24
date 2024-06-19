package client;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class ConsoleClient {
    final static int MY_TURN = 3;
    final static int OPPONENT_TURN = 4;
    private static final int WIN = 5;
    private static final int LOSE = 6;

    public static void main(String[] args) {
        try {
            // ユーザー入力を標準入力に設定。サーバーと接続
            Client client = new Client(new BufferedReader(
                    new InputStreamReader(System.in)));
            client.receiveAndLog(1); // 接続確認(サーバーに接続しました)を受け取る
            client.logging("新しく始めますか?(Yes/No)");
            boolean isNewAccount = client.yesNoInput();
            client.send(isNewAccount ? "new" : "login");
            if (!isNewAccount) {// ログインして始める
                while (true) {
                    client.logging("登録済みの名前を入力してください");
                    String inname = client.input("name");
                    client.logging("パスワードを入力してください");
                    String inpass = client.input("pass");
                    client.send(inname);
                    client.send(inpass);
                    String ack = client.receive();
                    if (ack.equals("correct")) {
                        client.logging("ログインしました");
                        break;
                    } else {
                        client.logging("名前またはパスワードが間違っています\n" +
                                "もう一度入力しますか?(Yes) 新規登録しますか?(No)");
                        if (client.yesNoInput()) {
                            client.send("onemore");
                            continue;
                        } else {
                            isNewAccount = true;
                            client.send("quit");
                            break;
                        }
                    }
                }
            }
            if (isNewAccount) {// 新規登録
                while (true) {
                    String name;
                    client.logging("新規の名前を入力してください");
                    name = client.input("name");
                    client.send(name);
                    if (client.receive().equals("duplicate")) {
                        client.logging("その名前は使用できません");
                    } else {
                        break;
                    }
                }
                while (true) {
                    String password1, password2;
                    client.logging("パスワードを入力してください");
                    password1 = client.input("pass");
                    client.logging("パスワードを再度入力してください");
                    password2 = client.input("pass");
                    if (password1.equals(password2)) {
                        client.send(password1);
                        client.logging("登録しました");
                        break;
                    } else {
                        client.logging("パスワードが一致しません\n" +
                                "1から入力し直してください");
                    }
                }
                client.logging("モンスターの属性を選択してください\n" +
                        "1: 火 2: 水 3: 草  4: 光  5: 闇");
                String monsterType = client.input("type");
                client.send(monsterType);
            }

            client.receiveAndLog(10);// モンスター情報を受け取
            client.logging("対戦相手を探しています");
            client.receiveAndLog(2); // 対戦相手を表示

            // 対戦
            if (client.receive().equals("first")) {
                client.logging("あなたが先攻です");
                client.setState(MY_TURN);
            } else {// receive() == "second"
                client.logging("あなたが後攻です");
                client.setState(OPPONENT_TURN);
            }

            // 対戦開始
            client.logging("対戦を開始します");
            while (client.getState() != WIN && client.getState() != LOSE) {
                if (client.getState() == MY_TURN) {
                    String moveIndex;
                    client.logging("あなたのターンです");
                    client.send(Integer.toString(client.getState()));// 先に状態を送る
                    client.receiveAndLog(10);// 技の表示
                    moveIndex = client.input("move");
                    client.send(Integer.toString(Integer.parseInt(moveIndex) - 1));
                    client.receiveAndLog(5 + 3);// 技の結果とHP表示
                    if (client.receive().equals("gameisover")) { // ゲーム終了判定
                        client.setState(WIN);
                        break;
                    } else {
                        client.setState(OPPONENT_TURN);
                    }
                } else if (client.getState() == OPPONENT_TURN) {
                    client.logging("相手のターンです");
                    client.send(Integer.toString(client.getState()));// 先に状態を送る
                    client.receiveAndLog(5 + 3);// 技の結果とHP表示
                    client.send("resultreceived"); // 技の結果の受け取りを報告
                    if (client.receive().equals("gameisover")) { // ゲーム終了判定
                        client.setState(LOSE);
                        break;
                    } else {
                        client.setState(MY_TURN);
                    }
                }
            }
            // 対戦結果の表示
            if (client.getState() == WIN) {
                client.logging("あなたの勝ちです");
            } else if (client.getState() == LOSE) {
                client.logging("あなたの負けです");
            }
            if(client.getState() == LOSE) {
                client.send("LOSE");
            }else if(client.getState() == WIN){
                client.send("WIN");
                client.receiveAndLog(10); // ステータスの表示
                client.logging("対戦に勝利したのでモンスターのステータスを1つ選んで強化することができます");
                client.logging("強化するステータスを選んでください\n1:hp 2:攻撃 3: 防御　4:特攻 5:特防 6:素早さ");
                String status = client.input("status");
                client.send(status);
                client.receiveAndLog(1 + 10); // 強化したステータス(1)とステータス全体の表示
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}