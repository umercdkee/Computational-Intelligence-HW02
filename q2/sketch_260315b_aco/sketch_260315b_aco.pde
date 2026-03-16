// ============================================================
//  ACO VISUALIZATION  Terrain Landscape with Pheromone Trails
// ============================================================

// Parameters
final int LAND_W = 1000;
final int LAND_H = 800;
final int PANEL_W = 200;

final int CELL = 5;
final int COLS = LAND_W / CELL;
final int ROWS = LAND_H / CELL;

int NUM_ANTS = 60;
float ALPHA = 0.5;
float BETA = 1.0;
float RHO = 0.05;
float Q = 5.0;
float MAX_PHER = 8.0;
float MIN_PHER = 0.01;

float[][] pher;
float[][] desirability;

int[] ax, ay;
float bestFit;
int bestCol, bestRow;

float[][] fitMap;
int[] bgPixels;
float minFit, maxFit;

int NUM_VALLEYS = 6;
float[] vcx = {150, 450, 850, 220, 750, 480};
float[] vcy = {150, 100, 320, 550, 650, 400};
float[] vdep = {0.9, 0.75, 0.85, 0.7, 0.8, 0.65};
float[] vwid = {95, 100, 85, 90, 90, 75};

final int HIST_LEN = 400;
float[] histBest = new float[HIST_LEN];
int histCount = 0;
boolean paused = false;
int iteration = 0;

final int[] DX = {-1, 0, 1, -1, 1, -1, 0, 1};
final int[] DY = {-1, -1, -1, 0, 0, 1, 1, 1};

void setup() {
  size(1200, 800);
  colorMode(RGB, 255);
  frameRate(60);

  computeLandscape();
  builddesrability();
  initPheromone();
  initAnts();
}

void draw() {
  loadPixels();
  for (int i = 0; i < bgPixels.length; i++) {
    pixels[i] = bgPixels[i];
  }
  for (int y = 0; y < LAND_H; y++) {
    for (int x = LAND_W; x < width; x++) {
      pixels[y * width + x] = color(248, 248, 245);
    }
  }
  updatePixels();

  drawPheromone();

  if (!paused) {
    updateACO();
    iteration++;
  }

  drawAnts();
  drawPanel();
}

float fitness(float lx, float ly) {
  float f = 1.0;
  for (int i = 0; i < NUM_VALLEYS; i++) {
    float dx = lx - vcx[i];
    float dy = ly - vcy[i];
    f -= vdep[i] * exp(-(dx * dx + dy * dy) / (2.0 * vwid[i] * vwid[i]));
  }
  f += 0.04 * sin(lx * 0.04) * cos(ly * 0.04);
  return f;
}

void computeLandscape() {
  fitMap = new float[LAND_W][LAND_H];
  bgPixels = new int[width * LAND_H];
  minFit = 1e9;
  maxFit = -1e9;

  for (int x = 0; x < LAND_W; x++) {
    for (int y = 0; y < LAND_H; y++) {
      float f = fitness(x, y);
      fitMap[x][y] = f;
      if (f < minFit) minFit = f;
      if (f > maxFit) maxFit = f;
    }
  }

  float range = maxFit - minFit;
  int NC = 14;
  float cStep = range / NC;

  for (int x = 0; x < LAND_W; x++) {
    for (int y = 0; y < LAND_H; y++) {
      float f = fitMap[x][y];
      float t = (f - minFit) / range;
      color c = terrainColor(t);

      if (x < LAND_W - 1 && y < LAND_H - 1) {
        float fr = fitMap[x + 1][y];
        float fd = fitMap[x][y + 1];
        for (int ci = 1; ci < NC; ci++) {
          float level = minFit + ci * cStep;
          if ((f - level) * (fr - level) < 0 || (f - level) * (fd - level) < 0) {
            c = blendColor(c, color(0), 0.35);
            break;
          }
        }
      }
      bgPixels[y * width + x] = c;
    }
  }
}

void builddesrability() {
  desirability = new float[COLS][ROWS];
  for (int c = 0; c < COLS; c++) {
    for (int r = 0; r < ROWS; r++) {
      float cx = c * CELL + CELL / 2.0;
      float cy = r * CELL + CELL / 2.0;
      float f = fitness(cx, cy);
      desirability[c][r] = 1.0 / max(f - minFit + 0.01, 0.01);
    }
  }
}

void initPheromone() {
  pher = new float[COLS][ROWS];
  for (int c = 0; c < COLS; c++) {
    for (int r = 0; r < ROWS; r++) {
      pher[c][r] = MIN_PHER;
    }
  }
}

void initAnts() {
  ax = new int[NUM_ANTS];
  ay = new int[NUM_ANTS];
  bestFit = 1e9;

  for (int i = 0; i < NUM_ANTS; i++) {
    ax[i] = (int) random(COLS);
    ay[i] = (int) random(ROWS);
    float f = fitness(ax[i] * CELL, ay[i] * CELL);
    if (f < bestFit) {
      bestFit = f;
      bestCol = ax[i];
      bestRow = ay[i];
    }
  }

  iteration = 0;
  histCount = 0;
  for (int i = 0; i < HIST_LEN; i++) {
    histBest[i] = 0;
  }
}

void updateACO() {
  for (int i = 0; i < NUM_ANTS; i++) {
    float[] scores = new float[8];
    float total = 0;

    for (int d = 0; d < 8; d++) {
      int nc = ax[i] + DX[d];
      int nr = ay[i] + DY[d];

      if (nc >= 0 && nc < COLS && nr >= 0 && nr < ROWS) {
        float tau = pow(pher[nc][nr], ALPHA);
        float eta = pow(desirability[nc][nr], BETA);
        scores[d] = tau * eta;
        total += scores[d];
      }
    }

    float pick = random(total);
    float curr_sum = 0;
    int chosen = -1;
    for (int d = 0; d < 8; d++) {
      curr_sum += scores[d];
      if (curr_sum >= pick && scores[d] > 0) {
        chosen = d;
        break;
      }
    }

    if (chosen != -1) {
      ax[i] += DX[chosen];
      ay[i] += DY[chosen];

      float f = fitness(ax[i] * CELL, ay[i] * CELL);
      float deposit = Q / max(f - minFit + 0.1, 0.1);
      pher[ax[i]][ay[i]] = min(pher[ax[i]][ay[i]] + deposit, MAX_PHER);

      if (f < bestFit) {
        bestFit = f;
        bestCol = ax[i];
        bestRow = ay[i];
      }
    }
  }

  // Evaporation
  for (int c = 0; c < COLS; c++) {
    for (int r = 0; r < ROWS; r++) {
      pher[c][r] = max(pher[c][r] * (1 - RHO), MIN_PHER);
    }
  }

  if (histCount < HIST_LEN) {
    histBest[histCount++] = bestFit;
  } else {
    for (int i = 0; i < HIST_LEN - 1; i++) {
      histBest[i] = histBest[i + 1];
    }
    histBest[HIST_LEN - 1] = bestFit;
  }
}

void drawPheromone() {
  noStroke();
  for (int c = 0; c < COLS; c++) {
    for (int r = 0; r < ROWS; r++) {
      float t = (pher[c][r] - MIN_PHER) / (MAX_PHER - MIN_PHER);
      if (t < 0.02) continue;

      color col = lerpColor(color(255, 200, 80, 0), color(220, 60, 10, 210), t);
      fill(col);
      rect(c * CELL, r * CELL, CELL, CELL);
    }
  }
}

void drawAnts() {
  noStroke();
  for (int i = 0; i < NUM_ANTS; i++) {
    float px = ax[i] * CELL + CELL / 2.0;
    float py = ay[i] * CELL + CELL / 2.0;

    fill(255, 255, 255, 180);
    ellipse(px, py, 5, 5);
    fill(20, 20, 20);
    ellipse(px, py, 3, 3);
  }

  float bx = bestCol * CELL + CELL / 2.0;
  float by = bestRow * CELL + CELL / 2.0;
  float pulse = 7 + 3 * sin(frameCount * 0.1);
  stroke(20, 20, 20);
  strokeWeight(2);
  noFill();
  ellipse(bx, by, pulse * 2.5, pulse * 2.5);
  noStroke();
  fill(20, 20, 20);
  ellipse(bx, by, 6, 6);
  fill(20, 20, 20);
  textSize(9);
  textAlign(LEFT);
  text("BEST", bx + 8, by - 5);
}

void drawPanel() {
  int ox = LAND_W + 14;
  textAlign(LEFT);

  fill(40, 40, 40);
  textSize(14);
  text("ACO Terrain", ox, 30);

  fill(60, 120, 60);
  textSize(11);
  text("Iteration: " + iteration, ox, 58);
  text("Best fit: " + nf(bestFit, 1, 5), ox, 74);
  text("Best pos: (" + bestCol * CELL + ", " + bestRow * CELL + ")", ox, 90);

  fill(80, 80, 80);
  textSize(10);
  text("Parameters ", ox, 116);
  text("Alpha: " + nf(ALPHA, 1, 2), ox, 132);
  text("Beta: " + nf(BETA, 1, 2), ox, 147);
  text("Evap: " + nf(RHO, 1, 3), ox, 162);
  text("Ants: " + NUM_ANTS, ox, 177);
  text("Valleys: " + NUM_VALLEYS, ox, 192);

  fill(120, 120, 120);
  textSize(9);
  text("CONTROLS", ox, 216);
  text("[SPACE]  Pause / Resume", ox, 230);
  text("[R]      Reset Simulation", ox, 243);
  text("[A / a]  Alpha (Pheromone) +/-", ox, 256);
  text("[B / b]  Beta (Desirability) +/-", ox, 269);
  text("[E / e]  Evaporation Rate +/-", ox, 282);
  text("[P / p]  Ant Population +/-", ox, 295);
  text("[Click]  Add New Valley", ox, 308);

  fill(80, 80, 80);
  textSize(10);
  text("Convergence", ox, 332);

  int gx = ox, gy = 342, gw = 176, gh = 118;
  stroke(180, 180, 175);
  strokeWeight(1);
  noFill();
  rect(gx, gy, gw, gh);

  fill(140);
  textSize(8);
  text("iter: ", gx + gw - 28, gy + gh + 10);

  if (histCount > 1) {
    float gmin = 1e9, gmax = -1e9;
    int plotN = min(histCount, HIST_LEN);
    for (int i = 0; i < plotN; i++) {
      if (histBest[i] < gmin) gmin = histBest[i];
      if (histBest[i] > gmax) gmax = histBest[i];
    }

    stroke(200, 80, 20);
    strokeWeight(1.5);
    beginShape();
    for (int i = 0; i < plotN; i++) {
      float cx = gx + map(i, 0, plotN - 1, 0, gw);
      float cy = gy + gh - map(histBest[i], gmin, gmax, 4, gh - 4);
      vertex(cx, cy);
    }
    endShape();
    strokeWeight(1);

    fill(200, 80, 20);
    textSize(8);
    text(nf(gmin, 1, 3), gx + 2, gy + gh - 3);
  }

  int ly = 490, lx = ox + 4;
  noStroke();
  fill(255, 200, 80, 160);
  rect(lx - 4, ly - 8, 14, 10);
  fill(220, 60, 10, 210);
  rect(lx - 4, ly + 7, 14, 10);
  fill(20, 20, 20);
  ellipse(lx + 3, ly + 30, 5, 5);
  fill(20, 20, 20);
  stroke(20, 20, 20);
  strokeWeight(1.5);
  noFill();
  ellipse(lx + 3, ly + 47, 12, 12);
  noStroke();

  fill(100);
  textSize(9);
  text("Low pheromone", lx + 14, ly);
  text("High pheromone", lx + 14, ly + 15);
  text("Ant", lx + 14, ly + 33);
  text("Global best", lx + 14, ly + 50);

  if (paused) {
    fill(180, 60, 60);
    textSize(12);
    text("PAUSED", ox, 600);
  }
}

// add valley using mouse
void mousePressed() {
  if (mouseX < LAND_W) {
    NUM_VALLEYS++;
    vcx = append(vcx, (float) mouseX);
    vcy = append(vcy, (float) mouseY);
    vdep = append(vdep, 0.8);
    vwid = append(vwid, 65.0);
    computeLandscape();
    builddesrability();
  }
}

void keyPressed() {
  switch (key) {
    case ' ':
      paused = !paused;
      break;
    case 'r':
    case 'R':
      initPheromone();
      initAnts();
      break;
    case 'A':
      ALPHA = min(ALPHA + 0.1, 5.0);
      break;
    case 'a':
      ALPHA = max(ALPHA - 0.1, 0.1);
      break;
    case 'B':
      BETA = min(BETA + 0.1, 5.0);
      break;
    case 'b':
      BETA = max(BETA - 0.1, 0.1);
      break;
    case 'E':
      RHO = min(RHO + 0.005, 0.2);
      break;
    case 'e':
      RHO = max(RHO - 0.005, 0.005);
      break;
    case 'P':
      NUM_ANTS += 10;
      initPheromone();
      initAnts();
      break;
    case 'p':
      NUM_ANTS = max(10, NUM_ANTS - 10);
      initPheromone();
      initAnts();
      break;
  }
}

color terrainColor(float t) {
  if (t < 0.15) return lerpColor(color(10, 60, 20), color(40, 110, 40), t / 0.15);
  if (t < 0.35) return lerpColor(color(40, 110, 40), color(90, 160, 55), (t - 0.15) / 0.20);
  if (t < 0.55) return lerpColor(color(90, 160, 55), color(180, 165, 100), (t - 0.35) / 0.20);
  if (t < 0.75) return lerpColor(color(180, 165, 100), color(145, 110, 75), (t - 0.55) / 0.20);
  return lerpColor(color(145, 110, 75), color(240, 238, 240), (t - 0.75) / 0.25);
}

color blendColor(color c1, color c2, float t) {
  return color(
    lerp(red(c1), red(c2), t),
    lerp(green(c1), green(c2), t),
    lerp(blue(c1), blue(c2), t)
  );
}
