package software.Action;

import java.io.*;
import java.util.Random;


public class Move extends Action{
    public static final int STRUGGLE = 50;
    public int damage;

    public Move(String name, int damage, int count, int type_value,boolean phy){
        this.name = name;
        this.damage = damage;
        this.count = count;
        this.type_value = type_value;
        this.phy = phy;

    }

    @Override
    public String toString(){
        String s1 = null, s2 = null;
        switch(type_value){
            case 1:
                s1 = "火";
                break;
            case 2:
                s1 = "水";
                break;
            case 3:
                s1 = "草";
                break;
            case 4:
                s1 = "光";
                break;
            case 5:
                s1 = "闇";
                break;
        }
        if(this.phy) s2 = "物理";
        if(!this.phy) s2 = "特殊";
        return ("\n\n技名: " + this.name + "\t威力: " + this.damage + "\t最大使用可能回数: " +this.count + "\t属性: " + s1 + "\t技種類: " + s2);
    }

    @Override
    public int Do(PrintWriter out) {
        if(this.count > 0){
            out.println(this.name);
            this.count --;
            return damage;
        }else{
            out.println("悪あがき");
            Random rand = new Random();
            return rand.nextInt(STRUGGLE);
        }
    }


    
}
