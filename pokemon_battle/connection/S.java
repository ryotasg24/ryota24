package software.connection;

import java.io.*;
import java.net.*;


public class S {
    public static final int PORT = 8080;//ポート番号を設定する
    final int MAXPALYER = 2;//
    public static void main(String[] args) throws IOException {
        ServerSocket s = new ServerSocket(PORT);// ソケットを作成する
        System.out.println("Started: " + s);
        SocketIO socketIO[] = new SocketIO[MAXPALYER];
        Monster monster[] = new Monster[MAXPALYER];
        for (int i = 0; i < MAXPALYER; i++) {
            socketIO[i] = new SocketIO(s.accept());
            System.out.println("Accept connection: " + socketIO[i].getSocket());
            monster[i] = new Monster(socketIO[i].getIn(), socketIO[i].getOut());
            monster[i].showStatus(socketIO[i].getOut());
            monster[i].showCurrent(socketIO[i].getOut());
        }
        Action act1 = new Action("death", 1, true, 60);
        String result = act1.exec(monster[0], monster[1]);
        for (int i = 0; i < MAXPALYER; i++) {
            socketIO[i].send(result);
            monster[i].showCurrent(socketIO[i].getOut());
            socketIO[i].send("END");
        }
        System.out.println("End of action");

        // close sockets
        for (SocketIO s : socketIO) {
            s.close();
        }
        s.close();
    }
}
