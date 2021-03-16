package AI.Components;

public class Weight {

    private double changeSum = 0;
    private int totalChanges = 0;
    private double value;

    public Weight(double value) {
        this.value = value;
    }

    public void addChange(double value) {
        this.changeSum += value;
        this.totalChanges++;
    }

    public void clearChanges() {
        this.changeSum = 0;
        this.totalChanges = 0;
    }

    public void applyChanges(double learningRate) {
        this.value -= changeSum / totalChanges * learningRate;
    }

    public double getValue() {
        return this.value;
    }
}
