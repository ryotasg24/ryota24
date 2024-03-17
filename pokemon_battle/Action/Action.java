package software.Action;

import java.io.PrintWriter;


abstract public class Action {
    public String name;
    public int count;
    public int type_value;
    public boolean phy;//物理か特殊か

    public boolean getPhy(){
        return phy;
    }

    public int getZokuseiValue(){
        return this.type_value;
    }
    public abstract int Do(PrintWriter out);
}
