package client;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;
import java.awt.event.*;
import javax.swing.*;

public class GraphicalClient {
    final static int MY_TURN = 3;
    final static int OPPONENT_TURN = 4;
    private static final int WIN = 5;
    private static final int LOSE = 6;

    public static final String[] MONSTER_TYPE = { "火", "水", "草", "光", "闇" };
    public static final String[] MONSTER_STATUS = { "HP", "攻撃", "防御", "特攻", "特防", "素早さ" };

    private static ClientUI gClient;
    private static Client cClient;

    private static boolean waitMoveSelect = true;
    private static int selectedMoveIndex = -1;

    private static String alliesName = "allies";
    private static String enemyName = "enemy";

    private static void logging(String str) {
        gClient.logging(str);
    }

    private static void warning(String msg) {
        gClient.warning(msg);
    }

    private static String input(String question) {
        return gClient.inputStr(question);
    }

    private static int optionInput(String question, String[] options) {
        return gClient.inputOption(question, options);
    }

    private static boolean yesNoInput(String question) {
        return gClient.inputYesNo(question);
    }

    private static String[] receiveAndLog(int line) throws IOException {
        String[] result = new String[line];
        for (int i = 0; i < line; i++) {
            result[i] = cClient.receive();
            logging(result[i]);
        }
        return result;
    }

    public static void main(String[] args) {
        // クライアントが強制終了(Ctrl+C)した時に起動するスレッド(サーバー側にSIGINTを送信
        Runtime.getRuntime().addShutdownHook(
            new Thread (() -> {cClient.send("SIGINT"); cClient.out = null; cClient.in = null;}));
        // initialize gui
        gClient = new ClientUI();
        String[] testWaza = { "waza1", "waza2", "waza3", "waza4" };
        gClient.setMove(testWaza);
        gClient.setImage(-1, "client/image/background.png");
        gClient.setImage(0, "client/image/question.png");
        gClient.setHP(0, 10, 10);
        gClient.setButtonAction(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                selectedMoveIndex = gClient.getSelectedMove();
                if (selectedMoveIndex != -1) {
                    waitMoveSelect = false;
                    // JOptionPane.showMessageDialog(null, gClient.getSelectedMove());
                }
            }
        });
        gClient.clearLog();

        gClient.setHP(1, 10, 10);
        gClient.setImage(1, "client/image/question.png");

        try {
            // ユーザー入力を標準入力に設定。サーバーと接続
            cClient = new Client(new BufferedReader(
                    new InputStreamReader(System.in)));
            receiveAndLog(1); // 接続確認応答を受け取る
            // logging("新しく始めますか?(Yes/No)");
            boolean isNewAccount = yesNoInput("新しく始めますか?");
            cClient.send(isNewAccount ? "new" : "login");
            if (!isNewAccount) {// ログインして始める
                while (true) {
                    String inname = input("登録済みの名前を入力してください");
                    logging(inname);
                    String inpass = input("パスワードを入力してください");
                    logging(inpass);
                    alliesName = inname;
                    cClient.send(inname);
                    cClient.send(inpass);
                    String ack = cClient.receive();
                    if (ack.equals("correct")) {
                        logging("ログインしました");
                        break;
                    } else {
                        warning("名前またはパスワードが間違っています\n");
                        if (yesNoInput(
                                "もう一度入力しますか?(Yes) 新規登録しますか?(No)")) {
                            cClient.send("onemore");
                            continue;
                        } else {
                            isNewAccount = true;
                            cClient.send("quit");
                            break;
                        }
                    }
                }
            }
            else if (isNewAccount) {// 新規登録
                while (true) {
                    String name;
                    name = input("新規の名前を入力してください");
                    alliesName = name;
                    logging(name);
                    cClient.send(name);
                    if (cClient.receive().equals("duplicate")) {
                        warning("その名前は使用できません");
                    } else {
                        break;
                    }
                }
                while (true) {
                    String password1, password2;
                    password1 = input("パスワードを入力してください");
                    logging(password1);
                    password2 = input("パスワードを再度入力してください");
                    logging(password2);
                    if (password1.equals(password2)) {
                        cClient.send(password1);
                        logging("登録しました");
                        break;
                    } else {
                        warning("パスワードが一致しません\n" +
                                "1から入力し直してください");
                    }
                }
                int typeIndex = optionInput("モンスターの属性を選択してください",
                        MONSTER_TYPE);
                logging("属性：" + MONSTER_TYPE[typeIndex]);
                // String monsterType = cClient.input("type");
                cClient.send(Integer.toString(typeIndex + 1));
            }
            // int id = Integer.parseInt(cClient.receive()); // id受け取る
            receiveAndLog(10);// モンスター情報を受け取る
            logging("対戦相手を探しています");
            String[] enemyInfo = receiveAndLog(2); // 対戦相手を表示
            enemyName = enemyInfo[1].substring(5, enemyInfo[1].length()-2);

            // 名前表示
            gClient.setStatus(alliesName);
            gClient.setEnemyStatus(enemyName);
            // 画像表示
            Random random = new Random();
            gClient.setImage(0, 
                String.format("client/image/allies/%d.png", 1+random.nextInt(21)));
            gClient.setImage(1, 
                String.format("client/image/enemy/%d.png", 1+random.nextInt(21)));
            gClient.setImage(0, "client/image/allies/"+ (Integer.valueOf(cClient.receive())%20)+".png");
            gClient.setImage(1, "client/image/enemy/"+ (Integer.valueOf(cClient.receive())%20)+".png");            
            // 対戦
            if (cClient.receive().equals("first")) {
                logging("あなたが先攻です");
                cClient.setState(MY_TURN);
            } else {// receive() == "second"
                logging("あなたが後攻です");
                cClient.setState(OPPONENT_TURN);
            }

            // 対戦相手が決まったらHPを受信する?
            int myMaxHp = Integer.parseInt(cClient.receive());
            int opponentMaxHp = Integer.parseInt(cClient.receive());
            gClient.setHP(0, myMaxHp, myMaxHp);
            gClient.setHP(1, opponentMaxHp, opponentMaxHp);

            // 対戦開始
            logging("対戦を開始します");
            while (cClient.getState() != WIN && cClient.getState() != LOSE) {
                if (cClient.getState() == MY_TURN) {
                    logging("あなたのターンです");
                    cClient.send(Integer.toString(cClient.getState()));// 先に状態を送る
                    String[] moveReceive = receiveAndLog(10);// 技の表示
                    String[] myMove = new String[4];
                    myMove[0] = moveReceive[1];
                    myMove[1] = moveReceive[3];
                    myMove[2] = moveReceive[5];
                    myMove[3] = moveReceive[7];
                    gClient.setMove(myMove);

                    waitMoveSelect = true;

                    while(waitMoveSelect){
                        try{
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            warning("sleep interrupted in wait move select");
                        }
                    }
                    // int moveIndex = optionInput("技を選択してください", myMove);
                    cClient.send(Integer.toString(selectedMoveIndex));

                    receiveAndLog(5 + 3);// 技の結果とHP表示
                    gClient.setHP(1, Integer.parseInt(cClient.receive()), opponentMaxHp); // 敵のHP更新
                    if (cClient.receive().equals("gameisover")) { // ゲーム終了判定
                        cClient.setState(WIN);
                        break;
                    } else {
                        cClient.setState(OPPONENT_TURN);
                    }
                } else if (cClient.getState() == OPPONENT_TURN) {
                    logging("相手のターンです");
                    cClient.send(Integer.toString(cClient.getState()));// 先に状態を送る
                    receiveAndLog(5 + 3);// 技の結果とHP表示
                    gClient.setHP(0, Integer.parseInt(cClient.receive()), myMaxHp); // 自分のHP更新
                    cClient.send("resultreceived"); // 技の結果の受け取りを報告
                    if (cClient.receive().equals("gameisover")) { // ゲーム終了判定
                        cClient.setState(LOSE);
                        break;
                    } else {
                        cClient.setState(MY_TURN);
                    }
                }
            }
            // 対戦結果の表示
            if (cClient.getState() == WIN) {
                logging("あなたの勝ちです");
                warning("あなたの勝ちです");
            } else if (cClient.getState() == LOSE) {
                logging("あなたの負けです");
                warning("あなたの負けです");
            }
            if (cClient.getState() == LOSE) {
                cClient.send("LOSE");
            } else if (cClient.getState() == WIN) {
                cClient.send("WIN");
                receiveAndLog(10);
                // logging("対戦に勝利したのでモンスターのステータスを1つ選んで強化することができます");
                // logging("強化するステータスを選んでください\n1:hp 2:攻撃 3: 防御 4:特攻 5:特防 6:素早さ");
                int status = optionInput("対戦に勝利したのでモンスターのステータスを1つ選んで強化することができます", MONSTER_STATUS);
                cClient.send(Integer.toString(status+1));
                receiveAndLog(1 + 10);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}