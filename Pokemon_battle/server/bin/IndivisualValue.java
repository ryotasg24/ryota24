package server.bin;

import java.util.Random;

public class IndivisualValue {

    private static int MAX_VALUE = 32;
    private String name;
    private int value;

    public IndivisualValue(String name){
        this.name = name;
        this.value = generateValue();
    }


    // 最小値10でステータス生成。 10になるのは大体6%

    private int generateValue(){
        
        int gaussian = (int) (new Random().nextGaussian() * 4 + 16);

        if(gaussian > MAX_VALUE)
            return MAX_VALUE;

        if(gaussian < 10)
            return 10;

        return gaussian;
    }

    public int getValue(){
        return this.value;
    }

    public int setValue(int value){
        return this.value = value;
    }
    
    public String showStats(){

        if(this.value < 1)
            return "ダメかも";

        if(this.value < 16)
            return "よくない";

        if(this.value < 26)
            return "悪くない";

        if(this.value < 31)
            return "良い";

        return "素晴らしい";
    }
}
