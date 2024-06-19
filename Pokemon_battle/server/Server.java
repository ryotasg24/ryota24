package server;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

import server.bin.IndivisualValue;
import server.bin.Monster;
import server.bin.Move;


public class Server extends Thread {

    private static final int PORT = 8080;
    static int MAX_CONNECTIONS = 128;

    ServerSocket serverSocket;
    ClientInfo clientsInfo [] = new ClientInfo[MAX_CONNECTIONS]; 
    Move [] importedMoves;

    // 対戦待ちの人のリスト

    List<Integer> waitingPlayers = new ArrayList<Integer>();// 対戦待ちリスト

    // 登録最中の人のリスト、サーバに名前(アカウント)が登録されるのはパスワードを設定してからなので
    // パスワード設定中に他の人が同じ名前で登録してしまわないようにこのリストを使う

    List<String> registeList = new ArrayList<String>();


    public static void main(String [] args) 
        throws IOException { new Server(); }

    public Server(){
        this.start();
    }

    public void run(){

        try {
        
            // サーバー起動中にも技を更新したり、クライアントの情報を表示したりできるようにするスレッド

            new Thread(() -> {

                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                
                while(true){
                    try {

                        // 入力があったらその内容に応じて処理(例えば、技を更新する)

                        if(reader.ready()){ 
                            String line = reader.readLine();
                            debug(line);
                        }
                        
                        Thread.sleep(1000);
                    
                    } catch (InterruptedException | IOException e){
                        e.printStackTrace();
                    }
                }
            }).start();

            serverSocket = new ServerSocket(PORT);

            System.out.println("サーバーが起動しました\n");
            
            importedMoves = fetchMovesFromCsv("./server/bin/moves.csv");
            
            while(true){
                Socket socket = serverSocket.accept();
                new Channel(socket,this);
            }

        } catch ( IOException e ){
            System.out.println("サーバーエラー");
        } finally {

            if(serverSocket == null)
                return;
                
            try {
                serverSocket.close();
            } catch ( IOException e ){
                e.printStackTrace();
            }
        }
    }


    // サーバー側でクライアントの状態を確認したり、技を更新したりする用
    
    private void debug(String line){

        if(line.equals("renewmoves")){
            importedMoves = fetchMovesFromCsv("./server/bin/moves.csv");
            return;
        }

        if(line.equals("client")){
            
            for(int i = 0; i < clientsInfo.length; i++)
                if(clientsInfo[i] != null){
                    System.out.println("id: " + i + " name: " + clientsInfo[i].name + 
                    (clientsInfo[i].online? " online": " offline"));
                }

            return;
        }

        if(line.equals("waiting")){
            
            System.out.println(waitingPlayers.size() + "人待ち");
            
            for(int i = 0;i < waitingPlayers.size();i++)
                System.out.println(i + "番目 " + clientsInfo[/*id = */waitingPlayers.get(i)].name);

            return;
        }


        System.out.println("他の文字列を入力してください");
        System.out.println("renewmoves: 技を更新 " + 
            "client: クライアントの一覧を表示 " + 
            "waiting: 対戦待ちの一覧を表示");
    }


    /*
     *  確認関連
     */
    
     // 名前が重複していたらtrueを返す
    
     public boolean isDuplicationName(String name){

        for(int i = 0;i < MAX_CONNECTIONS;i++)
            if(clientsInfo[i] != null && clientsInfo[i].name.equals(name))
                return true;

        for(int i = 0;i < registeList.size();i++)
            if(registeList.get(i).equals(name))
                return true;

        return false;
    }


    public boolean isGameover(int myId,int opponentId){
        return clientsInfo[opponentId].monster.hp < 1
            || clientsInfo[myId]      .monster.hp < 1 ;
    }


    // 登録した名前が存在してパスワードが正しいかったらtrueを返す

    public boolean isValidAccount(String name,String password) 
        throws NoSuchAlgorithmException {
        
        byte [] hashedPassword = 
            ClientInfo.getHashedPass(password);
        
        for(int i = 0;i < MAX_CONNECTIONS;i++){

            ClientInfo info = clientsInfo[i];

            if(info == null)
                continue;

            if(info.online)
                continue;

            if(!info.name.equals(name))
                continue;

       
            byte [] storedPassword = info.getPassward();
            
            
            if(storedPassword.length != hashedPassword.length)
                return false;

            for(int j = 0;j < hashedPassword.length;j++)
                if(storedPassword[j] != hashedPassword[j])
                    return false;

            return true;
        }

        return false;
    }


    // speedの値に応じて自分が先攻ならtrue, 後攻ならfalseを返す
    
    public boolean isFirstTurn(int myId,int opponentId){

        int 
            opponentSpeed = clientsInfo[opponentId].monster.speed.getValue() ,
            mySpeed = clientsInfo[myId].monster.speed.getValue() ;

        if(mySpeed > opponentSpeed)
            return true;
        
        if(mySpeed < opponentSpeed)
            return false;

        //mySpeed=opponentSpeedのとき

        if(myId < opponentId)
            return true;

        return false;
    }


    public String getNameById(int id){
        return clientsInfo[id].name;
    }


    public int getIdByName(String name){
        
        for(int i = 0;i < MAX_CONNECTIONS;i++)
            if(clientsInfo[i] != null && clientsInfo[i].name.equals(name))
                return i;
        
        return -1;
    }


    /*
     * 登録関連
     */
    
    // 新しいアカウントを登録し、アカウントのidを返す
    
    public int addAccount(
        String name , String plainPassword , 
        String monsterType , PrintWriter out
    ) throws NoSuchAlgorithmException {
        
        int i;
        
        for(i = 0;i < MAX_CONNECTIONS;i++)
            if(clientsInfo[i] == null){

                // 技とモンスターはランダムに生成

                var moves = chooseFourMoves(importedMoves);
                var monster = new Monster(moves,monsterType);

                clientsInfo[i] = new ClientInfo(i,name,plainPassword,monster,out);

                break;
            }

        return i;
    }

    
    // csvファイルmoves.csvから全ての技リストを読み込む(サーバー起動時に呼び出し)
    
    public Move[] fetchMovesFromCsv(String filename){
        
        int i = 0;
        
        try {

            var moves = new ArrayList<Move>();
            
            try(BufferedReader br = new BufferedReader(new FileReader(filename,StandardCharsets.UTF_8))){

                String line;
                
                while((line = br.readLine()) != null){

                    var moveInfo = line.split(",");
                    
                    if(moveInfo.length > 5){

                        var move = new Move(
                            moveInfo[0],
                            Integer.parseInt(moveInfo[1]),
                            Integer.parseInt(moveInfo[2]),
                            Integer.parseInt(moveInfo[3]),
                            moveInfo[4].equals("1") ? true : false,
                            Integer.parseInt(moveInfo[5])
                        );

                        moves.add(move); //命中率追加

                        i++;
                    }
                }
            }

            logging(i + "個の技を読み込みました");
            
            return moves.toArray(new Move[moves.size()]);
        } catch (IOException e) {
            System.out.println("for debug" +
                    "moves.csvのパスを確認してください\n" +
                    "(MacOS12.0.01で動作確認済み)\n" +
                    "パスはChannel.java内で指定しています\n");
        }

        return null;
    }


    // 4つのランダムに選んだ技を返す(モンスター生成時に呼び出し)

    public Move[] chooseFourMoves(Move[] moves) {
    
        // 実例: moves.length = 5 のとき[0,1,2,3,4]をシャッフルして[2,3,1,4,0]
        // のようになる。このときmoves[2], moves[3], moves[1], moves[4]のリストを返す。
    
        var list = new ArrayList<Integer>();
    
        for(int i = 0;i < moves.length;i++)
            list.add(i);

        for(int i = 0;i < 30;i++)
            Collections.shuffle(list);
        
        var returnMove = new Move[4];

        for(int i = 0;i < 4;i++)
            returnMove[i] = moves[list.get(i)];

        return returnMove;
    }


    // 相手を探して、相手のidを返す

    public int findOpponent(int id) 
        throws InterruptedException {
    
        // 自分の前に対戦待ちが1人以下かつ、自分の後ろか前に対戦待ちがいたらその人と対戦
    
        int index = waitingPlayers.indexOf(id);

        if(index > 1)
            return -1;
            
        if(waitingPlayers.size() < 2)
            return -1;

        index = 1 - index;
            
        return waitingPlayers.get(index);
    }


    // 対戦開始前に対戦開始待ちリストから削除する

    public void removeWaitingPlayer(int id){
        
        if(waitingPlayers.indexOf(id) != 0)
            return;
        
        waitingPlayers.remove(1);
        waitingPlayers.remove(0);
    }


    // プレイヤーが強制終了した時に呼び出される

    public void forceLogout(int id,int opponentId){

        if(waitingPlayers.contains(id)){
            System.out.println("forceLogout");
            waitingPlayers.remove(waitingPlayers.indexOf(id));
        }

        var self = clientsInfo[id];
        
        self.logout();
        self.monster.sigintPunish();
        
        if(opponentId == -1)
            return;
    
        var opponent = clientsInfo[opponentId];

        opponent.send("相手が強制終了しました");
        opponent.logout();
    }


    // 対戦の時にHPを送る

    public void sendHp(int myId, int idForHp){
    
        var info = clientsInfo[idForHp];

        // HPがマイナスにならないようにする

        int hp = Math.min(info.monster.hp,0);  
        
        info.send(Integer.toString(hp));
    }


    // ステータスを選んで3上げる 32が最大
    
    public void levelUp(int id,int status){

        var monster = clientsInfo[id].monster;

        Function<Integer,Integer> process = 
            (value) -> Math.min(value + 3,32);

        IndivisualValue property = null;

        switch(status){
        case 1 : property = monster.health  ; break ;
        case 2 : property = monster.attack  ; break ;
        case 3 : property = monster.block   ; break ;
        case 4 : property = monster.contact ; break ;
        case 5 : property = monster.defense ; break ;
        case 6 : property = monster.speed   ; break ;
        }

        if(property != null)
           property.setValue(process.apply(property.getValue()));
    }


    /*
     * バトル関連
     */

    public void showStats(int id){

        var info = clientsInfo[id];

        info.send(info.monster.toString());
    }


    public void showMoveLineup(int id){

        var info = clientsInfo[id];

        String str = "技は4種類あります\n";
        
        for(int i = 1;i < 5;i++)
            str += "技" + i + ": " + info.monster.moves[i - 1].toString() + "\n";
        
        str += "使いたい技を選択してください";

        info.send(str);
    }


    // 攻撃後にクライアント全員に攻撃結果を表示 moveIdxはクライアントが選択した技の番号

    public void useMove(int myId,int opponentId,int moveIdx){

        var opponent = clientsInfo[opponentId];
        var self = clientsInfo[myId];

        var oppMonster = opponent.monster;
        var myMonster = opponent.monster;
    
        String opponentName = opponent.name;
        String myName = self.name;
    
        Move myMove = myMonster.moves[moveIdx];
    
        String partition = "--------------";
        String result;


        // 選んだ技の使用可能回数が0なら悪あがきを繰り出す
        
        if(myMove.count <= 0){
            
            int damage = myMove.getStruggleGamage();
            
            oppMonster.hp -= damage;
            myMonster.hp -= damage / 2;
            
            result = 
                partition + "\n" +
                myName + "は" +
                myMove.name + "を使用できません\n" +
                "代わりに悪あがきを繰り出した!\n" +
                opponentName + "に" +
                damage + "のダメージを与えたが自分も" + damage / 2 + "のダメージを受けた\n" +
                partition;
        
            self.send(result);
            opponent.send(result);

            return;
        }


        // 技が命中しなかった場合
        
        if((100 - myMove.hitRate) > Math.random() * 100){

            // 命中しなくてもPP減らす

            myMove.count--;           
            
            result = 
                partition + "\n" +
                myName + "は" + myMove.name + "を使用した!" + "\n" +
                "しかし" + opponentName + "には当たらなかった!" + "\n" + 
                partition + "\n";
                
            self.send(result);
            opponent.send(result);

            return;
        }

        
        String compatibility;
        
        double multiplier = myMove.calculateMultiplier(myMonster,oppMonster);
        
        if(multiplier < 1.0){
            compatibility = "効果はいまいちだ\n";
        } else
        if(multiplier < 1.5){
            compatibility = "効果はまあまあだ\n";
        } else {
            compatibility = "効果は抜群だ\n";
        }

        int proccessedDamage = (int) (myMove.damage * multiplier);
        
        myMove.count--;
        oppMonster.hp -= proccessedDamage;
        
        result = 
            partition + "\n" + 
            myName + "は" + myMove.name + "を使用した!" + "\n" +
            compatibility + opponentName + "に" + proccessedDamage + "のダメージを与えた!" + "\n" + 
            partition;
        
        self.send(result);
        opponent.send(result);
    }


    public void showCurrentHp(int myId,int opponentId){
    
        var opponent = clientsInfo[opponentId];
        var self = clientsInfo[myId];

        int 
            opponentHp = Math.max(opponent.monster.hp,0) ,
            myHp = Math.max(self.monster.hp,0) ;
    
        String str = 
            "現在の" + self.name     + "のHPは " + myHp       + "です" + "\n" +
            "現在の" + opponent.name + "のHPは " + opponentHp + "です" + "\n" + 
            "-------------";
    
        self.send(str);
        opponent.send(str);
    }


    public void logging(String string){
        System.out.println(string);
    }
}
