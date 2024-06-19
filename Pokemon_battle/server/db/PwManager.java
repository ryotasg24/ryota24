package server.db;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;


public class PwManager {

    private final String 
        SQL_USER = "root" ,
        SQL_URL = "jdbc:mysql://localhost:3306/pockemonishgame" ;
    
    private String SQL_PW = "";

    private Map<String,String> pwlist = new HashMap<>();

    Statement statement;


    public PwManager(){

        if(SQL_PW.equals("")){
            Scanner sc = new Scanner(System.in);
            System.out.println("SQL PW");
            SQL_PW = sc.next();
            sc.close();
        }
        
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            Connection connection = DriverManager.getConnection(SQL_URL,SQL_USER,SQL_PW);
            this.statement = connection.createStatement();
            System.out.println("Database connected");
        } catch (Exception e) {
            System.out.println("Database connection error");
            e.printStackTrace();
        }
    }


    public boolean reloadPlayers(){

        String sql = "SELECT * FROM players";
        
        try {
            
            ResultSet rs = statement.executeQuery(sql);
            
            pwlist.clear();

            while(rs.next()){

                String 
                    name = rs.getString("name") ,
                    pw = rs.getString("pw") ;

                pwlist.put(name,pw);
            }

            return true;
        
        } catch (Exception e) {
            System.err.println("cannot reload players");
            e.printStackTrace();
            return false;
        }
    }


    public boolean addPlayer(String name,String pw){
    
        String sql = "INSERT INTO players (name, pw) VALUES('%s', '%s');";
        
        sql = String.format(sql,name,pw);
    
        try {
            statement.executeUpdate(sql);
        } catch (Exception e) {
            System.err.println("cannot add player");
            e.printStackTrace();
            return false;
        }
    
        pwlist.put(name,pw);
    
        return true;
    }


    public boolean addPlayer(Player player){
        return this.addPlayer(player.name,player.pw);
    }

    public boolean check(String name,String password){
        return pwlist.get(name).equals(password);
    }

    public String getPW(String name){
        return pwlist.get(name);
    }
}
