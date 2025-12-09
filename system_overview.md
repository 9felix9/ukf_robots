# ROS2 Homework – System Overview

Dieses Dokument beschreibt die Gesamtarchitektur des ROS2-Projekts bestehend aus:

- `fake_robot_node` (Simulationsnode)
- `kalman_positioning_node` (UKF-Lokalisierungsnode)

Es zeigt die Datenflüsse, verwendeten Topics sowie die Aufgaben jedes Nodes.

---

# 1. Nodes im Workspace

Im Workspace existieren zwei aktive ROS2-Nodes:

## **/fake_robot_node**
Simuliert die Welt und erzeugt alle Sensor- und Ground-Truth-Daten.

- Hat *keine* Subscriber
- Publiziert mehrere Topics (sowohl Ground Truth als auch verrauschte Messungen)

## **/kalman_positioning_node**
Implementiert den Unscented Kalman Filter (UKF), der die Roboterpose aus verrauschten Daten rekonstruiert.

- Abonniert simulierte Sensoren
- Berechnet eine Schätzung des Roboterzustands
- Publiziert die geschätzte Odometry

---

# 2. Topics und ihre Funktionen

Folgende Topics existieren im System (Ausgabe von `ros2 topic list -t`):

| Topic | Typ | Bedeutung |
|-------|------|-----------|
| `/robot_gt` | `nav_msgs/msg/Odometry` | Wahre Roboterpose (Ground Truth) |
| `/robot_noisy` | `nav_msgs/msg/Odometry` | Verrauschte Odometry (Eingang für UKF-Prediction) |
| `/robot_estimated_odometry` | `nav_msgs/msg/Odometry` | Vom UKF berechnete Schätzung der Roboterpose |
| `/landmarks_gt` | `sensor_msgs/msg/PointCloud2` | Wahre Landmark-Positionen |
| `/landmarks_observed` | `sensor_msgs/msg/PointCloud2` | Verrauschte Landmark-Messungen (Eingang für UKF-Update) |
| `/tf` | `tf2_msgs/msg/TFMessage` | Transformationen der Simulation |
| `/rosout` | `rcl_interfaces/msg/Log` | ROS2-Logging |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | Parameteränderungen |

---

# 3. Node-Kommunikation

## **/fake_robot_node**

### Publisher:
- `/robot_gt`
- `/robot_noisy`
- `/landmarks_gt`
- `/landmarks_observed`
- `/tf`
- `/parameter_events`
- `/rosout`

### Subscriber:
- keine

→ **Funktion:**  
Erzeugt die komplette simulierte Sensorik und Ground Truth.

---

## **/kalman_positioning_node**

### Subscriber:
- `/robot_noisy`  
  → wird für den **Prediction Step** des UKF verwendet  
- `/landmarks_observed`  
  → wird für den **Update Step** des UKF verwendet

### Publisher:
- `/robot_estimated_odometry`  
  → finale, geschätzte Pose des Roboters
- `/parameter_events`
- `/rosout`

→ **Funktion:**  
Führt den UKF-Lokalisierungsalgorithmus aus.

---

# 4. Datenfluss im System

```
                +---------------------+
                |   fake_robot_node   |
                |  (Simulation)        |
                +---------------------+
                   |            |
      Ground Truth |            |   Simulierte Sensorik
                   v            v
         /robot_gt            /robot_noisy  ---> UKF Prediction
         /landmarks_gt       /landmarks_observed ---> UKF Update

                +---------------------------+
                |   kalman_positioning_node |
                |           (UKF)           |
                +---------------------------+
                           |
                           v
               /robot_estimated_odometry
```

---

# 5. Was bedeutet „Ground Truth“?

**Ground Truth = die echte, perfekte Realität**, wie sie in der Simulation existiert.

- `/robot_gt` → perfekte Roboterpose  
- `/landmarks_gt` → perfekte Landmark-Karte  

Ground Truth dient nur zum Vergleichen/Visualisieren.

---

# 6. Nützliche ROS2-Befehle

### Node-Liste anzeigen
```
ros2 node list
```

### Informationen zu einem Node
```
ros2 node info <node_name>
```

### Topics anzeigen
```
ros2 topic list -t
```

### Topic-Inhalt anzeigen
```
ros2 topic echo /robot_estimated_odometry
```

---

# 7. Zusammenfassung

✔ `fake_robot_node` simuliert Sensordaten und die echte Welt  
✔ `kalman_positioning_node` schätzt die Roboterpose mit einem UKF  
✔ Sensor-Inputs: `/robot_noisy`, `/landmarks_observed`  
✔ Output: `/robot_estimated_odometry`  
✔ Ground Truth dient dem Debugging und Evaluieren des UKF
