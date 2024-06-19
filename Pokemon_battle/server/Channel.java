package server;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.security.NoSuchAlgorithmException;

import server.bin.Monster;


public class Channel extends Thread {

	BufferedReader in;
	PrintWriter out;

	Server server;
	Socket socket;

	public int 
		opponentId = -1 , 
		id = -1 ;

	// クライアントの状態。ターンの判定などに使う

	private final String 
		OPPONENT_TURN = "4" ,
		MY_TURN = "3" ;


	Channel(Socket socket,Server server){
		
		this.server = server;
		this.socket = socket;
		
		try {

			in = new BufferedReader(
				new InputStreamReader(socket.getInputStream()));
			
			out = new PrintWriter(
				socket.getOutputStream(),true);

		} catch (IOException e) {
			System.out.println("入出力接続エラー");
		}

		start();
	}


	// クライアントの1回の対戦に対応
	
	public void run(){

		try {

			send("サーバーに接続しました");

			boolean isNewAccount = 
				receive().equals("new");

			// ログイン

			if(!isNewAccount)
				while(true){
					
					String 
						name = receive() ,
						password = receive() ;
					
					if(server.isValidAccount(name,password)){
					
						send("correct");
					
						id = server.getIdByName(name);
					
						server.clientsInfo[id].login(out);
					
						break;
					} else {

						send("incorrect");
						
						if(receive().equals("onemore"))
							continue;
						
						isNewAccount = true;
						break;
					}
				}


			// 新規登録

			if(isNewAccount){ 
			
				String name , password , monsterType;
			
				while(true){

					name = receive();
					
					if(server.isDuplicationName(name)){
						send("duplicate");
					} else {

						send("correct");
						
						// 登録中の人のリストに名前を追加

						if(!server.registeList.contains(name))
							server.registeList.add(name);

							break;
					}
				}

				password = receive();
				monsterType = receive();
				
				if(server.registeList.contains(name))
					server.registeList.remove(server.registeList.indexOf(name));

				id = server.addAccount(name,password,monsterType,out);
			}

			// send(Integer.toString(id)); // id送る
			
			server.showStats(id);
			server.waitingPlayers.add(id);

			while(opponentId == -1){
			
				// 対戦待ちでSIGINTを受信する用

				if(in.ready())
					receive();

				opponentId = server.findOpponent(id);
				Thread.sleep(100);
			}

			String opponentName = server
				.getNameById(opponentId);

			Thread.sleep(500);
			server.removeWaitingPlayer(id);

			send("対戦相手が見つかりました" + "\n" 
				+ "対戦相手は" + opponentName + "です");
			
			send(String.valueOf(id));
			send(String.valueOf(opponentId));
			
			send(server.isFirstTurn(id,opponentId) 
				? "first" 
				: "second");


			// 対戦開始
			// 対戦前にHP送信

			server.sendHp(id,/* id for hp = */id);
			server.sendHp(id,opponentId);

			while(true){

				String clientState = receive();
				
				if(clientState.equals(MY_TURN)){

					server.showMoveLineup(id);
					
					int moveIndex = Integer.parseInt(receive());
					
					server.useMove(id,opponentId,moveIndex);
					server.showCurrentHp(id,opponentId);
					server.sendHp(id,opponentId);
					server.sendHp(opponentId,opponentId);

					if(server.isGameover(id,opponentId)){
						send("gameisover");
						break;
					}
					
					send(OPPONENT_TURN);

					continue;
				}


				// 攻撃側の呼び出しで技の結果, HPを表示

				if(clientState.equals(OPPONENT_TURN)){
					
					// 技の結果の受け取り、HPの受け取り確認

					receive();

					if(server.isGameover(id,opponentId)){
						send("gameisover");
						break;
					}

					send(MY_TURN);
				}
			}
			
			Thread.sleep(500);
			
			if(receive().equals("WIN")){

				server.showStats(id);
				
				int status = Integer.parseInt(receive());

				server.levelUp(id,status);
				
				send(Monster.val2stats(status) + "が強くなりました");
				
				server.showStats(id);
			}

		} catch ( IOException | InterruptedException | NoSuchAlgorithmException e ){
			System.out.println("クライアント切断");
		} finally {
			server.clientsInfo[id].logout();
		}
	}


	public void send(String value){
		out.println(value);
	}


	public String receive() throws IOException {

		String line = in.readLine();
		
		if(line.equals("SIGINT")){
			System.out.println("クライアント" + id + "が強制終了しました");
			server.forceLogout(id,opponentId);
		}

		return line;
	}


	public void logging(Object value){
		System.out.println(value);
	}
}