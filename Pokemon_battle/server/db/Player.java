package server.db;

import java.util.Random;


public class Player {

    // generated at https://randomwordgenerator.com/name.php

    private static String[] randomNames = {
        "Miles Blake" , "Nigel Bennett" , "Ora Rivera" , "Ericka Lyons" , 
        "Jacquelyn Drake" , "Betty Knapp" , "Shannon Oconnell" , "Chong Leon" , 
        "Darren Rocha" , "Wanda Hogan" , "James Chaney" , "Eileen West" , 
        "Cristobal Valenzuela" , "Kurtis Graves" , "Madeline Zuniga", "Len Jensen" , 
        "Lynda Mullen" , "Jon Hancock" , "Myron Boyle" , "Margarito Coffey",
    };

    static int count = 0;


    String name , pw ;
    int id; // Unused


    public Player(int id,String name,String pw){
        this.name = name;
        this.id = id;
        this.pw = pw;
        count++;
    }

    public Player(String name,String pw){
        this(count,name,pw);
    }
    

    public Player(String pw){
        
        this(null,pw);

        Random random = new Random();
        
        this.name = randomNames[random.nextInt(randomNames.length)];
    }

    public static void resetCount(){
        count = 0;
    }

    @Override
    public String toString(){
        return String.format("{id:%d, name:\"%s\", pw:\"%s\"}",id,name,pw);
    }
}
