import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import random
import importlib.util
import inspect
from deap import base, creator, tools, algorithms

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 默认策略
def default_a_strategy(history_a, history_b):
    return 50000  # A分给B 5万

def default_b_strategy(history_a, history_b):
    return random.choice([True, False])  # B随机接受或拒绝

def find_single_function(module):
    """查找并返回模块中的唯一函数"""
    functions = [func for func in dir(module) if callable(getattr(module, func)) and not func.startswith("__")]
    
    if len(functions) == 1:
        return getattr(module, functions[0])
    else:
        raise ValueError("上传的文件中应仅包含一个函数。")

def load_strategy(file_path, default_strategy):
    try:
        spec = importlib.util.spec_from_file_location("strategy", file_path)
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # 查找文件中的唯一函数
        strategy = find_single_function(strategy_module)
        
        # 验证函数参数是否符合要求
        if inspect.signature(strategy).parameters.keys() != {'history_a', 'history_b'}:
            print(f"函数的参数不符合要求，使用默认策略。")
            return default_strategy
        
        return strategy
    except Exception as e:
        print(f"加载策略文件时发生错误: {e}")
        return default_strategy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_a', methods=['POST'])
def upload_a():
    file_a = request.files.get('file_a', None)

    if file_a and file_a.filename != '':
        file_a_path = os.path.join(app.config['UPLOAD_FOLDER'], file_a.filename)
        file_a.save(file_a_path)
        session['a_strategy_path'] = file_a_path  # 存储文件路径到session
        with open(file_a_path, 'r') as f:
            code_a = f.read()
    else:
        session['a_strategy_path'] = None
        code_a = inspect.getsource(default_a_strategy)
    
    return jsonify({"code_a": code_a})

@app.route('/upload_b', methods=['POST'])
def upload_b():
    file_b = request.files.get('file_b', None)

    if file_b and file_b.filename != '':
        file_b_path = os.path.join(app.config['UPLOAD_FOLDER'], file_b.filename)
        file_b.save(file_b_path)
        session['b_strategy_path'] = file_b_path  # 存储文件路径到session
        with open(file_b_path, 'r', encoding="utf-8") as f:
            code_b = f.read()
    else:
        session['b_strategy_path'] = None
        code_b = inspect.getsource(default_b_strategy)
    
    return jsonify({"code_b": code_b})

@app.route('/tournament')
def tournament():
    a_strategies = []
    b_strategies = []

    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if filename.endswith('_a.py'):
            a_strategies.append((filename, load_strategy(file_path, default_a_strategy)))
        elif filename.endswith('_b.py'):
            b_strategies.append((filename, load_strategy(file_path, default_b_strategy)))

    results = []

    # 进行两两对决，运行 100 次并计算平均结果
    for a_name, a_strategy in a_strategies:
        total_gain = 0
        total_accepted = 0
        total_rounds = 0
        player_name = a_name[:-5]
        if 'default' in player_name: player_name = player_name.replace("default", "baseline")

        # 运行 100 次模拟
        n = 100

        for b_name, b_strategy in b_strategies:
            for _ in range(n):
                history_a = []
                history_b = []
                accepted = False
                final_offer = 0

                for round_number in range(100):
                    offer = a_strategy(history_a, history_b)
                    history_a.append(offer)
                    accept = b_strategy(history_a, history_b)
                    history_b.append(accept)

                    if accept:
                        accepted = True
                        final_offer = offer if offer > 0 else 0
                        break

                total_gain += (100000 - final_offer) if accepted else 0
                total_accepted += 1 if accepted else 0
                total_rounds += round_number

        # 计算平均报价和接受率
        average_gain = total_gain / n / len(b_strategies)
        acceptance_rate = total_accepted / n / len(b_strategies)

        results.append({
            "player_name": player_name,
            "a_name": a_name,
            "average_gain": average_gain,
            "acceptance_rate": acceptance_rate,
            "total_rounds": total_rounds / n / len(b_strategies)  # 平均回合数
        })    

    # 按平均报价排序
    results = sorted(results, key=lambda x: (x['average_gain']), reverse=True)

    return render_template('tournament.html', results=results)

@app.route('/train')
def run_train():
    a_strategies = []
    b_strategies = []

    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if filename.endswith('_a.py'):
            a_strategies.append((filename, load_strategy(file_path, default_a_strategy)))
        elif filename.endswith('_b.py'):
            b_strategies.append((filename, load_strategy(file_path, default_b_strategy)))

    player_name = "wi_evil"

    # 运行 100 次模拟
    n = 100

    def evaluate(individual):
        total_gain = 0
        total_accepted = 0
        total_rounds = 0

        for b_name, b_strategy in b_strategies:
            for _ in range(n):
                history_a = []
                history_b = []
                accepted = False
                final_offer = 0

                for round_number in range(100):
                    offer = individual[round_number]
                    history_a.append(offer)
                    accept = b_strategy(history_a, history_b)
                    history_b.append(accept)

                    if accept:
                        accepted = True
                        final_offer = offer if offer > 0 else 0
                        break

                total_gain += (100000 - final_offer) if accepted else 0
                total_accepted += 1 if accepted else 0
                total_rounds += round_number

        # 计算平均报价和接受率
        average_gain = total_gain / n / len(b_strategies)

        print(average_gain)

        return (average_gain,)

    random.seed(42)

    # 创建适应度和个体类
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 适应度类，用于最大化适应度
    creator.create("Individual", list, fitness=creator.FitnessMax)  # 个体类，包含策略数组和适应度

    # 定义工具箱，用于初始化种群和定义遗传操作
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 60000)  # 定义属性生成函数，生成0到50000之间的整数
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=100)  # 定义个体生成函数，生成长度为100的策略数组
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 定义种群生成函数，生成包含多个个体的种群

    # 注册评估函数
    toolbox.register("evaluate", evaluate)

    # 定义遗传算法的操作
    toolbox.register("mate", tools.cxTwoPoint)  # 定义交叉操作，使用两点交叉
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=100000, indpb=0.2)  # 定义变异操作，使用均匀整数变异
    toolbox.register("select", tools.selTournament, tournsize=3)  # 定义选择操作，使用锦标赛选择

    # 定义运行遗传算法的函数
    def run_genetic_algorithm():
        """
        运行遗传算法，寻找最优策略。
        返回:
            tuple: 包含最优种群和日志的元组。
        """

        population = toolbox.population(n=500)  # 创建初始种群，包含50个个体
        ngen = 200  # 定义遗传算法的代数
        cxpb, mutpb = 0.5, 0.2  # 定义交叉概率和变异概率

        # 定义统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x)/len(x))  # 平均适应度
        stats.register("min", min)  # 最小适应度
        stats.register("max", max)  # 最大适应度

        # 运行遗传算法
        pop, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)  # 使用简单的进化算法

        return pop, log

    def run_genetic_algorithm_with_elitism():
        # 运行算法并获取最佳策略
        elite_size = 5  # 精英个体数量

        population = toolbox.population(n=50)  # 创建初始种群，包含50个个体
        ngen = 20  # 定义遗传算法的代数
        cxpb, mutpb = 0.5, 0.2  # 定义交叉概率和变异概率

        # 记录历史最优个体
        best_individual = None
        best_fitness = float('-inf')

        for gen in range(ngen):
            fitnesses = map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                if isinstance(fit, tuple) and len(fit) > 0:
                    ind.fitness.values = fit
                else:
                    ind.fitness.values = (0,)  # 设置一个默认值

            # 更新历史最优个体
            for ind in population:
                if ind.fitness.values[0] > best_fitness:
                   best_fitness = ind.fitness.values[0]
                   best_individual = ind

            # 选择、交叉和变异
            offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
            population = offspring + [best_individual]
            population = toolbox.select(population, k=len(population) - elite_size)

        print(best_fitness)
        return best_individual

    best_strategy = run_genetic_algorithm_with_elitism()

    # best_strategy, _ = run_genetic_algorithm()
    # best_strategy = best_strategy[0]  # 获取最佳个体

    print("Best Strategy:", best_strategy)
    # print(stats)

    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    a_strategy_path = session.get('a_strategy_path', None)
    b_strategy_path = session.get('b_strategy_path', None)

    a_strategy = load_strategy(a_strategy_path, default_a_strategy) if a_strategy_path else default_a_strategy
    b_strategy = load_strategy(b_strategy_path, default_b_strategy) if b_strategy_path else default_b_strategy

    # 模拟运行逻辑
    history_a = []
    history_b = []
    accepted = False
    final_offer = 0
    
    for round_number in range(100):
        offer = a_strategy(history_a, history_b)  # 调用A策略
        history_a.append(offer)
        
        accept = b_strategy(history_a, history_b)  # 调用B策略
        history_b.append(accept)
        
        if accept:
            accepted = True
            final_offer = offer
            break
    
    # 返回模拟结果
    return jsonify({
        "round_number": round_number,
        "final_offer": final_offer,
        "accepted": accepted,
        "history_a": history_a,
        "history_b": history_b
    })

if __name__ == '__main__':
    app.run(debug=True)
