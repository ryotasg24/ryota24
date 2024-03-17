package software.connection;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

public class A {
    public static void main(String[] args) throws IOException{
        ServerSocket s = new ServerSocket(80);
        for (int i = 0; i < 3; i++) {
            Socket socket = s.accept();
            System.out.println(i + "accept");
            System.out.println(socket);
        }
        s.close();
    }
}