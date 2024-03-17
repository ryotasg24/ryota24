package software.Action;

import java.io.*;
import java.util.Random;


public class Heal extends Action{
    public static final int PRAYER = 30;
    public int healing;

    public Heal(String name, int healing, int type_value){
        this.name = name;
        this.healing = healing;
        this.type_value = type_value;
        this.count = 15;
        this.phy = false;

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
        return ("\n\n技名: " + this.name + "\t威力: " + this.healing + "\t最大使用可能回数: " +this.count + "\t属性: " + s1 + "\t技種類: " + s2);
    }

    @Override
    public int Do(PrintWriter out) {
        if(this.count > 0){
            out.println(this.name);
            this.count --;
            return -healing;
        }else{
            out.println("祈り");
            Random rand = new Random();
            return -rand.nextInt(PRAYER);
        }
    }


    
}
