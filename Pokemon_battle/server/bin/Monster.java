package server.bin;

public class Monster {

    public IndivisualValue 
        contact , // 特攻
        defense , // 特防
        health  , // hp
        attack  , // 攻撃
        speed   , // 素早さ
        block   ; // 防御

    public String 
        name , // 名前
        type ; // 属性

    public Move moves[]; // 技; 

    public int 
        maxHp , // 実hp
        sum   , // 個体値
        hp    ;


    public Monster(Move moves [],String type){
        
        this.contact = new IndivisualValue("特攻");
        this.defense = new IndivisualValue("特防");
        this.attack = new IndivisualValue("攻撃");
        this.health = new IndivisualValue("HP");
        this.block = new IndivisualValue("防御");
        this.speed = new IndivisualValue("素早さ");
        
        this.moves = moves;
        this.type = type;

        resetHp();
    }
    

    public static int type2val(String type){

        switch(type){
        default   : return 0 ;
        case "火" : return 1 ;
        case "水" : return 2 ;
        case "草" : return 3 ;
        case "光" : return 4 ;
        case "闇" : return 5;
        }
    }


    public static String val2type(int value){
        
        switch(value){
        default : return "" ;
        case  1 : return "火" ;
        case  2 : return "水" ;
        case  3 : return "草" ;
        case  4 : return "光" ;
        case  5 : return "闇" ;
        }
    }


    public static String val2stats(int value){

        switch(value){
        default : return "" ;
        case  1 : return "hp値" ;
        case  2 : return "攻撃" ;
        case  3 : return "防御" ;
        case  4 : return "特攻" ;
        case  5 : return "特防" ;
        case  6 : return "素早さ" ;
        }
    }

    public static String sum2evaluation(int sum){
        
        if(sum > 130)  
            return "素晴らしい個体!";

        if(sum > 105)
           return "相当優秀な個体です。";
        
        //上位50％

        if(sum > 93)
           return "平均以上な個体です。";    
            
        // 下位50％

        return "まずまずな個体です。";
    }


    public String getType(){
        return this.type;
    }


    @Override
    public String toString(){

        String partition = "---------------";
        String evaluation = sum2evaluation(sum);

        // Rem: 10行

        return partition 
            + "\n" + "属性は " + val2type(Integer.parseInt(getType())) 
            + "\n" + "hp値は " + health.getValue() 
            + "\n" + "攻撃は " + attack.getValue() 
            + "\n" + "防御は " + block.getValue() 
            + "\n" + "特攻は" + contact.getValue() 
            + "\n" + "特防は " + defense.getValue() 
            + "\n" + "素早さは " + speed.getValue() 
            + "\n" + "総合評価: " + evaluation 
            + "\n" + partition ;
    }


    public void decreaseHp(int damage){
        this.hp -= damage;
    }


    public void resetHp(){

        this.sum 
            = this.contact.getValue() 
            + this.defense.getValue() 
            + this.health.getValue() 
            + this.attack.getValue() 
            + this.block.getValue() 
            + this.speed.getValue() ;
        
        this.maxHp 
            = this.sum * 2 
            + this.health.getValue() 
            + 110 ;
        
        this.hp = this.maxHp;
    }


    // 強制終了したら罰を課す(負けそうになったら抜けれてしまうから)
    
    public void sigintPunish(){
        
        this.defense.setValue(this.defense.getValue() / 2);
        this.contact.setValue(this.contact.getValue() / 2);
        this.health.setValue(this.health.getValue() / 2);
        this.attack.setValue(this.attack.getValue() / 2);
        this.block.setValue(this.block.getValue() / 2);
        this.speed.setValue(this.speed.getValue() / 2);
        
        resetHp();
    }
}
