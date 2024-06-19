package server;

import java.io.PrintWriter;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import server.bin.Monster;


public class ClientInfo {

    private PrintWriter out;
    private byte[] hashedPassword;

    public Monster monster;
    public String name;
    public int id;

    boolean online = false;


    ClientInfo(
        int id , String name , String plainPassword , 
        Monster monster, PrintWriter out
    ) throws NoSuchAlgorithmException {
    
        this.hashedPassword = getHashedPass(plainPassword);
        this.monster = monster;
        this.online = true; //ログイン判定で使うかも
        this.name = name;
        this.out = out;
        this.id = id;
    }


    public byte [] getPassward(){
        return hashedPassword;
    }


    // ログインした時に呼び出す
    
    public void login(PrintWriter out){

        this.online = true;
        
        monster.resetHp();
        
        for(int i = 0;i < 4;i++)
            monster.moves[i].reset();

        this.out = out;
    } 
    

    // 対戦(通信)終了時に呼び出す

    public void logout(){
    
        out = null;
    
        if(monster != null){
    
            monster.resetHp();
    
            for(int i = 0;i < monster.moves.length;i++)
                monster.moves[i].reset();
        }

        online = false;
    }


    public void send(String message){
        
        if(out == null)
            return;

        out.println(message);
    }

    public static byte [] getHashedPass(String plainText)
        throws NoSuchAlgorithmException {
        
        MessageDigest SHA256 = 
            MessageDigest.getInstance("SHA-256");
        
        byte [] bytes = 
            plainText.getBytes();
        
        return SHA256.digest(bytes);
    }
}
