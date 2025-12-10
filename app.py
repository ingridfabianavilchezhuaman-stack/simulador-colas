# app.py
"""
Simulador educativo de colas M/M/1 y M/M/c
- Analítico: M/M/1 y M/M/c (Erlang C)
- Simulación real: SimPy (llegadas Poisson, servicios exponenciales)
- Interfaz: Streamlit con pestañas (Modelo, Simulación, Interpretación, Exportar)
- Producto pensado para uso docente (registros, representaciones, preguntas guiadas)
"""

import streamlit as st
import numpy as np
import pandas as pd
import simpy
import random
import math
import plotly.express as px
from io import StringIO

st.set_page_config(layout="wide", page_title="Simulador Colas M/M/1 y M/M/c")

# ---------------------------
# Funciones analíticas
# ---------------------------

def mm1_analitico(lmbda, mu):
    """M/M/1 analítico (devuelve dict)."""
    if lmbda >= mu:
        return {"estable": False}
    rho = lmbda / mu
    L = rho / (1 - rho)
    Lq = rho**2 / (1 - rho)
    W = 1 / (mu - lmbda)
    Wq = lmbda / (mu * (mu - lmbda))
    return {"estable": True, "rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq}

def mmc_analitico(lmbda, mu, c):
    """
    M/M/c analítico usando Erlang C.
    - a = lambda / mu
    - rho = lambda / (c * mu)
    Returns dict with 'estable' flag and metrics.
    """
    if c < 1:
        raise ValueError("c debe ser >= 1")
    if lmbda >= c * mu:
        return {"estable": False}
    a = lmbda / mu
    rho = lmbda / (c * mu)  # utilización por sistema

    # calcular P0
    sum_terms = sum((a**n) / math.factorial(n) for n in range(c))
    last = (a**c) / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (sum_terms + last)

    # Erlang C
    ErlangC = (a**c / math.factorial(c)) * (1 / (1 - rho)) * P0

    # Lq, Wq, W, L
    Lq = (ErlangC * lmbda) / (c * mu - lmbda)
    Wq = Lq / lmbda
    W = Wq + 1.0 / mu
    L = lmbda * W

    return {"estable": True, "a": a, "rho": rho, "P0": P0, "ErlangC": ErlangC,
            "Lq": Lq, "Wq": Wq, "W": W, "L": L}

# ---------------------------
# Simulación SimPy (M/M/c)
# ---------------------------

def customer_process(env, name, server, mu, stats):
    """Proceso de un cliente individual."""
    arrival = env.now
    with server.request() as req:
        yield req
        wait = env.now - arrival
        stats["waits"].append(wait)
        service_time = random.expovariate(mu)
        yield env.timeout(service_time)
        stats["sojourns"].append(env.now - arrival)

def arrival_generator(env, server, lmbda, mu, stats, tiempo_max):
    """Generador de llegadas Poisson hasta tiempo_max."""
    i = 0
    while env.now < tiempo_max:
        i += 1
        inter = random.expovariate(lmbda)
        yield env.timeout(inter)
        env.process(customer_process(env, f"c{i}", server, mu, stats))

def run_simulation_once(lmbda, mu, c, tiempo_max, seed=None):
    """Corre una simulación y devuelve estadísticas (listas)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=c)
    stats = {"waits": [], "sojourns": []}
    env.process(arrival_generator(env, server, lmbda, mu, stats, tiempo_max))
    env.run(until=tiempo_max)
    return stats

def run_simulations(lmbda, mu, c, tiempo_max, replications=1, base_seed=42):
    """Corre 'replications' simulaciones y devuelve promedios y lista de resultados por réplica."""
    all_results = []
    for r in range(replications):
        seed = base_seed + r
        stats = run_simulation_once(lmbda, mu, c, tiempo_max, seed=seed)
        avg_wait = float(np.mean(stats["waits"])) if stats["waits"] else 0.0
        avg_sojourn = float(np.mean(stats["sojourns"])) if stats["sojourns"] else 0.0
        n_served = len(stats["sojourns"])
        all_results.append({"rep": r+1, "avg_wait": avg_wait, "avg_sojourn": avg_sojourn, "n_served": n_served, "raw": stats})
    # compute aggregated stats
    df = pd.DataFrame([{"rep": r["rep"], "avg_wait": r["avg_wait"], "avg_sojourn": r["avg_sojourn"], "n_served": r["n_served"]} for r in all_results])
    summary = {"mean_wait": df["avg_wait"].mean(), "mean_sojourn": df["avg_sojourn"].mean(), "total_served": df["n_served"].sum(), "df_reps": df}
    return summary, all_results

# ---------------------------
# UI - Layout y controles
# ---------------------------

st.title("Simulación REAL de Colas M/M/1 y M/M/c (Prototipo educativo)")

tab1, tab2, tab3, tab4 = st.tabs(["Modelo", "Simulación", "Interpretación didáctica", "Exportar resultados"])

with tab1:
    st.header("Parámetros del modelo")
    col1, col2, col3 = st.columns(3)
    with col1:
        lmbda = st.number_input("λ — tasa de llegada (arrivals/unit)", value=0.9, min_value=0.01, step=0.01, format="%.4f")
        mu = st.number_input("μ — tasa de servicio por servidor (services/unit)", value=1.0, min_value=0.01, step=0.01, format="%.4f")
    with col2:
        c = st.slider("c — número de servidores", min_value=1, max_value=20, value=2)
        st.markdown("**Modelo seleccionado:** M/M/c (Poisson arrivals, exponential service, c servidores, FCFS)")
    with col3:
        tiempo_max = st.number_input("Tiempo de simulación (tiempo total)", min_value=100, max_value=200000, value=5000, step=100)
        replications = st.number_input("Replicaciones (simulaciones independientes)", min_value=1, max_value=50, value=3, step=1)

    st.markdown("---")
    st.subheader("Resultados analíticos (fórmulas)")

    if c == 1:
        ana = mm1_analitico(lmbda, mu)
    else:
        ana = mmc_analitico(lmbda, mu, c)

    if not ana.get("estable", True):
        st.error("⚠ El sistema NO es estable (λ ≥ c·μ). Las métricas tienden a infinito.")
    else:
        st.json(ana)

with tab2:
    st.header("Simulación real (SimPy)")
    st.markdown("Ajusta parámetros en la pestaña 'Modelo', luego haz clic en 'Correr simulación'.")

    colA, colB = st.columns([1,2])
    with colA:
        if st.button("Correr simulación REAL"):
            with st.spinner("Simulando... esto puede tardar según el tiempo y las repeticiones..."):
                summary, all_results = run_simulations(lmbda, mu, c, tiempo_max, replications=replications)
            st.success("Simulación completada")
            # mostrar resumen
            st.subheader("Resumen agregado de simulaciones")
            st.write(f"- Promedio del tiempo de espera (Wq) empírico (promedio de réplicas): {summary['mean_wait']:.4f}")
            st.write(f"- Promedio del tiempo total en sistema (W) empírico (promedio de réplicas): {summary['mean_sojourn']:.4f}")
            st.write(f"- Clientes atendidos (suma sobre réplicas): {summary['total_served']}")
            # detalle por réplica
            st.subheader("Detalle por réplica")
            st.dataframe(summary["df_reps"])

            # gráficos: histograma de esperas (usar la primera réplica como ejemplo)
            first_raw = all_results[0]["raw"]
            if first_raw["waits"]:
                fig = px.histogram(first_raw["waits"], nbins=50, title="Histograma de tiempos de espera (réplica 1)")
                fig.update_layout(xaxis_title="Tiempo de espera", yaxis_title="Frecuencia")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hubo clientes atendidos en la réplica 1 (prueba con tiempos de simulación mayores).")

            # comparación analítico vs simulado
            if ana.get("estable", True):
                st.subheader("Comparación ANALÍTICA vs SIMULADA")
                w_analitica = ana.get("W", None)
                wq_analitica = ana.get("Wq", None)
                st.write("Analítico (W):", w_analitica)
                st.write("Analítico (Wq):", wq_analitica)
                st.write("Simulado (W promedio):", summary["mean_sojourn"])
                st.write("Simulado (Wq promedio):", summary["mean_wait"])
                # pequeña tabla comparativa
                comp_df = pd.DataFrame([
                    {"métrica": "W (total)", "analítico": w_analitica, "simulado_promedio": summary["mean_sojourn"]},
                    {"métrica": "Wq (cola)", "analítico": wq_analitica, "simulado_promedio": summary["mean_wait"]}
                ])
                st.table(comp_df)
            # guardar en sesión para exportar
            st.session_state["last_sim_summary"] = summary
            st.session_state["last_sim_all"] = all_results
        else:
            st.info("Presiona 'Correr simulación REAL' para iniciar la simulación con los parámetros actuales.")

with tab3:
    st.header("Interpretación didáctica y guías de trabajo")
    st.markdown("""
### 🔷 Representaciones
Integramos tres registros clave:
- **Simbólico**: las fórmulas analíticas (M/M/1 y M/M/c - Erlang C).
- **Numérico/Tabular**: resultados de las simulaciones reales.
- **Gráfico**: histogramas y comparaciones visuales.

Esto ayuda a la *coordinación de registros* (Duval): el estudiante confronta la fórmula con datos reales.

### 🔷 Aproximación al límite
Observa qué ocurre cuando la **utilización ρ** se acerca a 1:
- El tiempo promedio en cola crece muy rápido.
- Un pequeño aumento en λ provoca grandes aumentos en Wq.
- La simulación muestra la variabilidad real y la inestabilidad cuando λ ≈ c·μ.

### 🔷 Preguntas guiadas (actividad)
1. Fija μ y aumenta λ en pequeños pasos: ¿cómo cambian W y Wq?  
2. Compara 1 servidor vs 3 servidores manteniendo la misma λ: ¿qué mejora observas en W?  
3. ¿En qué situaciones convendría aumentar servidores vs aumentar la tasa de servicio?

### 🔷 Notas metodológicas
- La **simulación** refleja variabilidad; por eso ejecutamos varias réplicas y promediamos.
- Aumenta `tiempo de simulación` para reducir la varianza de estimadores empíricos.
- Cuando λ ≥ c·μ, el sistema es **inestable**: las métricas analíticas divergen y la simulación mostrará crecimientos continuos en la cola.
""")

with tab4:
    st.header("Exportar / Descargar resultados")

    if "last_sim_summary" not in st.session_state:
        st.info("No hay resultados simulados aún. Corre la simulación en la pestaña 'Simulación'.")
    else:
        summary = st.session_state["last_sim_summary"]
        all_results = st.session_state["last_sim_all"]

        # preparar CSV de réplicas
        df_reps = summary["df_reps"]
        csv = df_reps.to_csv(index=False)
        st.download_button("Descargar tabla de réplicas (CSV)", data=csv, file_name="replicas_simulacion.csv", mime="text/csv")

        # preparar CSV con datos de la primera réplica (ejemplo)
        first_raw = all_results[0]["raw"]
        if first_raw["waits"]:
            df_waits = pd.DataFrame({"waits": first_raw["waits"], "sojourns": first_raw["sojourns"]})
            csv2 = df_waits.to_csv(index=False)
            st.download_button("Descargar datos (réplica 1) - tiempos (CSV)", data=csv2, file_name="datos_replica1.csv", mime="text/csv")
        else:
            st.info("La réplica 1 no registró clientes atendidos; ajusta tiempo de simulación y vuelve a correr.")

    st.markdown("---")
    st.write("Sugerencia: incluye los CSV en tu informe para mostrar la comparación entre la teoría (Erlang C) y los datos empíricos (SimPy).")

# ---------------------------
# FIN
# ---------------------------



