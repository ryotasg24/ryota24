package client;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.border.*;

public class ClientUI extends JFrame {

    static final int PROGRESSBARHIGHT = 15, PROGRESSBARWIDTH = 145;
    JLabel backgroundImage, alliesImage, enemyImage;// 背景画像，味方の画像，敵の画像
    JLabel alliesHPlabel, alliesHP, enemyHPlabel, enemyHP;// 味方の「HP」の表示，味方のXX/YYの表示，以下敵の
    JLabel alliesName, enemyName;// モンスターの名前
    JLabel aliiesSquare, enemySquare;// ステータス表示の背景
    JProgressBar alliesHPbar, enemyHPbar;// HPバー
    JList<String> Movelist;
    JButton okButton;
    JTextArea logArea;

    public ClientUI() {
        setTitle("Pokemonish Game");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setBounds(450, 50, 812, 835);
        getContentPane().setLayout(null);

        backgroundImage = new JLabel("no image");
        backgroundImage.setBounds(0, 0, 800, 600);

        alliesImage = new JLabel("no image");
        alliesImage.setBounds(180, 420, 100, 100);
        alliesHP = new JLabel("100/100");
        alliesHP.setBounds(600, 485, 80, 20);
        alliesHPlabel = new JLabel("HP");
        alliesHPlabel.setBounds(480, 466, 200, 20);
        alliesHPbar = new JProgressBar(1, 100);
        alliesHPbar.setBounds(500, 470, PROGRESSBARWIDTH, PROGRESSBARHIGHT);
        alliesHPbar.setBackground(Color.GRAY);
        alliesHPbar.setForeground(Color.GREEN);
        alliesHPbar.setBorderPainted(false);
        alliesName = new JLabel("allies");
        alliesName.setBounds(453, 445, 200, 20);
        alliesName.setFont(new Font(Font.DIALOG, Font.BOLD, 20));
        aliiesSquare = new JLabel("");
        aliiesSquare.setBounds(450, 445, 200, 60);
        aliiesSquare.setOpaque(true);
        aliiesSquare.setBackground(Color.WHITE);
        aliiesSquare.setBorder(new LineBorder(Color.GRAY, 2, false));

        String[] initialList = { "Move1", "Move2", "Move3", "Move4" };
        Movelist = new JList<>(initialList);
        Movelist.setBounds(0, 705, 705, 95);
        Movelist.setFont(new Font(Font.DIALOG, Font.PLAIN, 15));
        Movelist.setBorder(new LineBorder(Color.ORANGE, 3, false));
        //2行にするなら
        //Movelist.setVisibleRowCount(2);
        //Movelist.setLayoutOrientation(JList.HORIZONTAL_WRAP);
        

        okButton = new JButton("OK");
        okButton.setBounds(705, 705, 95, 95);
        okButton.setBorder(new LineBorder(new Color(160, 200, 255), 3, false));

        logArea = new JTextArea("log start...\n", 5, 20);
        JScrollPane logScroll = new JScrollPane(logArea);
        logScroll.setBounds(0, 600, 800, 105);
        logScroll.setBorder(new LineBorder(Color.GRAY, 3, false));

        enemyImage = new JLabel("no image enemy");
        enemyImage.setBounds(540, 140, 100, 100);
        enemyHP = new JLabel("100/100");
        enemyHP.setBounds(375, 215, 80, 20);
        enemyHPlabel = new JLabel("HP");
        enemyHPlabel.setBounds(250, 196, 200, 20);
        enemyHPbar = new JProgressBar(1, 100);
        enemyHPbar.setBounds(270, 200, PROGRESSBARWIDTH, PROGRESSBARHIGHT);
        enemyHPbar.setBackground(Color.GRAY);
        enemyHPbar.setForeground(Color.GREEN);
        enemyHPbar.setBorderPainted(false);
        enemyName = new JLabel("enemy");
        enemyName.setBounds(223, 175, 200, 20);
        enemyName.setFont(new Font(Font.DIALOG, Font.BOLD, 20));
        enemySquare = new JLabel("");
        enemySquare.setBounds(220, 175, 200, 60);
        enemySquare.setOpaque(true);
        enemySquare.setBackground(Color.WHITE);
        enemySquare.setBorder(new LineBorder(Color.GRAY, 2, false));

        // TODO: set size of each component

        getContentPane().add(alliesImage);
        getContentPane().add(alliesHPlabel);
        getContentPane().add(alliesHPbar);
        getContentPane().add(alliesHP);
        getContentPane().add(alliesName);
        getContentPane().add(aliiesSquare);

        getContentPane().add(logScroll);
        getContentPane().add(Movelist);
        getContentPane().add(okButton);

        getContentPane().add(enemyImage);
        getContentPane().add(enemyHPlabel);
        getContentPane().add(enemyHPbar);
        getContentPane().add(enemyHP);
        getContentPane().add(enemyName);
        getContentPane().add(enemySquare);

        getContentPane().add(backgroundImage);
        setVisible(true);
    }

    // HPバー操作味方：sw0, 敵：sw1
    public void setHP(int sw, int hp, int max) {
        JProgressBar bar = null;
        JLabel label = null;
        switch (sw) {
            case 0:// 味方
                bar = this.alliesHPbar;
                label = this.alliesHP;
                break;
            case 1:
                bar = this.enemyHPbar;
                label = this.enemyHP;
                break;
        }

        int value = hp * 100 / max;
        if (hp <= 0)
            value = 0;
        if (value > 50) {
            bar.setForeground(Color.GREEN);
        } else if (value > 20) {
            bar.setForeground(Color.YELLOW);
        } else {
            bar.setForeground(Color.RED);
        }

        bar.setValue(value);
        label.setText(hp + "/" + max);
    }

    // 画像表示
    // sw: 背景-1, 味方0, 敵1
    public void setImage(int sw, String file) {// 画像操作
        JLabel label = null;
        switch (sw) {
            case -1:// 背景
                label = this.backgroundImage;
                break;
            case 0:// 味方
                label = this.alliesImage;
                break;
            case 1:// 敵
                label = this.enemyImage;
                break;
        }
        ImageIcon icon = new ImageIcon(file);
        label.setIcon(icon);
        label.setText("");
    }

    public void setStatus(String newStatus) {
        this.alliesName.setText(newStatus);
    }

    public void setMove(String[] newMove) {
        this.Movelist.setListData(newMove);
    }

    public void setButtonAction(ActionListener a) {
        this.okButton.addActionListener(a);
    }

    public void logging(String log) {
        this.logArea.append(log + "\n");
        this.logArea.setCaretPosition(this.logArea.getText().length());
    }

    public void clearLog() {
        this.logArea.setText(null);
    }

    public int getSelectedMove() {
        return Movelist.getSelectedIndex();
    }

    public void setEnemyStatus(String status) {
        this.enemyName.setText(status);
    }

    // Stringの入力を返す
    public String inputStr(String question) {
        String ans = null;
        while (ans == null) {
            ans = JOptionPane.showInputDialog(this, question, "Input", 
                JOptionPane.QUESTION_MESSAGE);
        }
        return ans;
    }

    // 選択肢の番号を返す
    public int inputOption(String question, String[] options) {
        // JOptionPane.showInputDialog(this, "msg", null, JOptionPane.ERROR_MESSAGE,
        //     null, options, null);
        int ans = JOptionPane.showOptionDialog(this, question, 
            "Option dialog", JOptionPane.OK_OPTION, 
            JOptionPane.QUESTION_MESSAGE, null, options, null);
        return ans;
    }

    // yes:0 no:1を返す
    public boolean inputYesNo(String question) {
        int ans = JOptionPane.showConfirmDialog(this, question, "Yes or No", 
            JOptionPane.YES_NO_OPTION);
        if (ans == JOptionPane.YES_OPTION) return true;
        else return false;
    }

    public void warning(String msg) {
        JOptionPane.showMessageDialog(this, msg, "Warning", JOptionPane.WARNING_MESSAGE);
    }
}