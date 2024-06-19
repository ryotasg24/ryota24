package server.bin;

import java.util.Random;


// TODO Healの追加

public class Move {

    public static final int STRUGGLE_DAMAGE = 50;
    
    public boolean isPhysical;
    public String name;

    public int 
        typeValue ,
        maxcount ,
        hitRate ,   // 命中率
        damage ,
        count ;

    
    public Move(
        String name , int damage , int maxcount ,
        int typeValue , boolean isPhysical , int hitRate
    ){
        this.isPhysical = isPhysical;
        this.typeValue = typeValue;
        this.maxcount = maxcount;
        this.hitRate = hitRate;  //命中率
        this.count = maxcount;
        this.damage = damage;
        this.name = name;
    }


    @Override
    public String toString(){

        String 
            s1 = Monster.val2type(typeValue) , 
            s2 = isPhysical ? "物理" : "特殊" ;
        
        return 
            "技名: " + name + " " +
            "威力: " + damage + " " +
            "使用可能回数: " + count + " " +
            "属性: " + s1 + " " +
            "技種類: " + s2 + " " +
            "命中率: " + hitRate + "\n";
    }


    public double calculateMultiplier(Monster myMonster,Monster oppMonster){

        final int 
            myType = Monster.type2val(myMonster.type) ,
            oppType = Monster.type2val(oppMonster.type) ;

        double 
            same = 1.0 , // タイプ一致
            you  = 1.0 , // 相手の光闇
            con  = 1.0 , // タイプ相性
            me   = 1.0 , // 自分の光闇
            k    = 1.0 ; // 物理特殊倍率

        if(myType == this.typeValue)
            same = 1.5;
        
        switch(myType){
        case 1:
            switch(typeValue){
            case 2 : con = 0.5 ; break ;
            case 3 : con = 2.0 ; break ;
            }
            break;
        case 2:
            switch(typeValue){
            case 3 : con = 0.5 ; break ;
            case 1 : con = 2.0 ; break ;
            }
            break;
        case 3:
            switch(typeValue){
            case 1 : con = 0.5 ; break ;
            case 2 : con = 2.0 ; break ;
            }
            break;
        }


        k = isPhysical
            ? (double) myMonster.attack.getValue() / oppMonster.block.getValue()
            : (double) myMonster.contact.getValue() / oppMonster.defense.getValue() ;


        // kは最大32で大きすぎると一発で試合が終わるので最大を2にする

        if(k > 2)
            k = 2; 


        switch(myType){
        case 4:
            me = 0.5;
            break;
        case 5:
            me = 2.0;
            break;
        }

        switch(oppType){
        case 4:
            you = 0.5;
            break;
        case 5:
            you = 2.0;
            break;
        }


        return same * con * k * me * you;
    }


    public int getStruggleGamage(){
        return new Random()
            .nextInt(STRUGGLE_DAMAGE);
    }


    public void reset(){
        this.count = this.maxcount;
    }
}
