class RiverCrossingChecker:
    def __init__(self, N, k, moves):
        self.N = N
        self.k = k
        self.moves = moves
        self.left_bank = set([f'a_{i+1}' for i in range(N)] + [f'A_{i+1}' for i in range(N)])
        self.right_bank = set()
        self.boat_side = 'left'
        self.failed_step = None
        self.failed_people = []

    def _validate_state(self):
        for side in [self.left_bank, self.right_bank]:
            actors = {p for p in side if p.startswith('a_')}
            agents = {p for p in side if p.startswith('A_')}
            for actor in actors:
                num = actor.split('_')[1]
                if f'A_{num}' not in agents and agents:
                    self.failed_people = [actor] + list(agents)
                    return False
        return True

    def _validate_move(self, move):
        boat_set = set(move)
        if len(boat_set) == 0 or len(boat_set) > self.k:
            self.failed_people = list(boat_set)
            return False
        current_side = self.left_bank if self.boat_side == 'left' else self.right_bank
        if not boat_set.issubset(current_side):
            self.failed_people = list(boat_set - current_side)
            return False
        return True

    def _apply_move(self, move):
        source = self.left_bank if self.boat_side == 'left' else self.right_bank
        destination = self.right_bank if self.boat_side == 'left' else self.left_bank
        for person in move:
            source.remove(person)
            destination.add(person)
        self.boat_side = 'right' if self.boat_side == 'left' else 'left'

    def check(self):
        """
        Returns True if the entire sequence of moves is valid and ends with everyone on the right bank.
        Otherwise returns False and stores error info in `self.failed_step` and `self.failed_people`.
        """
        self.left_bank = set([f'a_{i+1}' for i in range(self.N)] + [f'A_{i+1}' for i in range(self.N)])
        self.right_bank = set()
        self.boat_side = 'left'

        for i, move in enumerate(self.moves):
            if not self._validate_move(move):
                self.failed_step = i
                return False
            self._apply_move(move)
            if not self._validate_state():
                self.failed_step = i
                return False

        expected = set([f'a_{i+1}' for i in range(self.N)] + [f'A_{i+1}' for i in range(self.N)])
        return self.right_bank == expected


# Ejemplo de configuración
N = 100  # Número de actores/agentes
k = 4  # Capacidad de la barca

# Ejemplo de secuencia de movimientos válida para N=3 y k=2
moves =[['A_1', 'a_1', 'A_2', 'a_2'], ['A_1', 'a_1'], ['A_3', 'a_3', 'A_4', 'a_4'], ['A_2', 'a_2'], ['A_5', 'a_5', 'A_6', 'a_6'], ['A_3', 'a_3'], ['A_7', 'a_7', 'A_8', 'a_8'], ['A_4', 'a_4'], ['A_9', 'a_9', 'A_10', 'a_10'], ['A_5', 'a_5'], ['A_11', 'a_11', 'A_12', 'a_12'], ['A_6', 'a_6'], ['A_13', 'a_13', 'A_14', 'a_14'], ['A_7', 'a_7'], ['A_15', 'a_15', 'A_16', 'a_16'], ['A_8', 'a_8'], ['A_17', 'a_17', 'A_18', 'a_18'], ['A_9', 'a_9'], ['A_19', 'a_19', 'A_20', 'a_20'], ['A_10', 'a_10'], ['A_21', 'a_21', 'A_22', 'a_22'], ['A_11', 'a_11'], ['A_23', 'a_23', 'A_24', 'a_24'], ['A_12', 'a_12'], ['A_25', 'a_25', 'A_26', 'a_26'], ['A_13', 'a_13'], ['A_27', 'a_27', 'A_28', 'a_28'], ['A_14', 'a_14'], ['A_29', 'a_29', 'A_30', 'a_30'], ['A_15', 'a_15'], ['A_31', 'a_31', 'A_32', 'a_32'], ['A_16', 'a_16'], ['A_33', 'a_33', 'A_34', 'a_34'], ['A_17', 'a_17'], ['A_35', 'a_35', 'A_36', 'a_36'], ['A_18', 'a_18'], ['A_37', 'a_37', 'A_38', 'a_38'], ['A_19', 'a_19'], ['A_39', 'a_39', 'A_40', 'a_40'], ['A_20', 'a_20'], ['A_41', 'a_41', 'A_42', 'a_42'], ['A_21', 'a_21'], ['A_43', 'a_43', 'A_44', 'a_44'], ['A_22', 'a_22'], ['A_45', 'a_45', 'A_46', 'a_46'], ['A_23', 'a_23'], ['A_47', 'a_47', 'A_48', 'a_48'], ['A_24', 'a_24'], ['A_49', 'a_49', 'A_50', 'a_50'], ['A_25', 'a_25'], ['A_51', 'a_51', 'A_52', 'a_52'], ['A_26', 'a_26'], ['A_53', 'a_53', 'A_54', 'a_54'], ['A_27', 'a_27'], ['A_55', 'a_55', 'A_56', 'a_56'], ['A_28', 'a_28'], ['A_57', 'a_57', 'A_58', 'a_58'], ['A_29', 'a_29'], ['A_59', 'a_59', 'A_60', 'a_60'], ['A_30', 'a_30'], ['A_61', 'a_61', 'A_62', 'a_62'], ['A_31', 'a_31'], ['A_63', 'a_63', 'A_64', 'a_64'], ['A_32', 'a_32'], ['A_65', 'a_65', 'A_66', 'a_66'], ['A_33', 'a_33'], ['A_67', 'a_67', 'A_68', 'a_68'], ['A_34', 'a_34'], ['A_69', 'a_69', 'A_70', 'a_70'], ['A_35', 'a_35'], ['A_71', 'a_71', 'A_72', 'a_72'], ['A_36', 'a_36'], ['A_73', 'a_73', 'A_74', 'a_74'], ['A_37', 'a_37'], ['A_75', 'a_75', 'A_76', 'a_76'], ['A_38', 'a_38'], ['A_77', 'a_77', 'A_78', 'a_78'], ['A_39', 'a_39'], ['A_79', 'a_79', 'A_80', 'a_80'], ['A_40', 'a_40'], ['A_81', 'a_81', 'A_82', 'a_82'], ['A_41', 'a_41'], ['A_83', 'a_83', 'A_84', 'a_84'], ['A_42', 'a_42'], ['A_85', 'a_85', 'A_86', 'a_86'], ['A_43', 'a_43'], ['A_87', 'a_87', 'A_88', 'a_88'], ['A_44', 'a_44'], ['A_89', 'a_89', 'A_90', 'a_90'], ['A_45', 'a_45'], ['A_91', 'a_91', 'A_92', 'a_92'], ['A_46', 'a_46'], ['A_93', 'a_93', 'A_94', 'a_94'], ['A_47', 'a_47'], ['A_95', 'a_95', 'A_96', 'a_96'], ['A_48', 'a_48'], ['A_97', 'a_97', 'A_98', 'a_98'], ['A_49', 'a_49'], ['A_99', 'a_99', 'A_100', 'a_100'], ['A_50', 'a_50'], ['A_51', 'a_51', 'A_52', 'a_52'], ['A_53', 'a_53'], ['A_55', 'a_55', 'A_56', 'a_56'], ['A_54', 'a_54'], ['A_57', 'a_57', 'A_58', 'a_58'], ['A_55', 'a_55'], ['A_59', 'a_59', 'A_60', 'a_60'], ['A_56', 'a_56'], ['A_61', 'a_61', 'A_62', 'a_62'], ['A_57', 'a_57'], ['A_63', 'a_63', 'A_64', 'a_64'], ['A_58', 'a_58'], ['A_65', 'a_65', 'A_66', 'a_66'], ['A_59', 'a_59'], ['A_67', 'a_67', 'A_68', 'a_68'], ['A_60', 'a_60'], ['A_69', 'a_69', 'A_70', 'a_70'], ['A_61', 'a_61'], ['A_71', 'a_71', 'A_72', 'a_72'], ['A_62', 'a_62'], ['A_73', 'a_73', 'A_74', 'a_74'], ['A_63', 'a_63'], ['A_75', 'a_75', 'A_76', 'a_76'], ['A_64', 'a_64'], ['A_77', 'a_77', 'A_78', 'a_78'], ['A_65', 'a_65'], ['A_79', 'a_79', 'A_80', 'a_80'], ['A_66', 'a_66'], ['A_81', 'a_81', 'A_82', 'a_82'], ['A_67', 'a_67'], ['A_83', 'a_83', 'A_84', 'a_84'], ['A_68', 'a_68'], ['A_85', 'a_85', 'A_86', 'a_86'], ['A_69', 'a_69'], ['A_87', 'a_87', 'A_88', 'a_88'], ['A_70', 'a_70'], ['A_89', 'a_89', 'A_90', 'a_90'], ['A_71', 'a_71'], ['A_91', 'a_91', 'A_92', 'a_92'], ['A_72', 'a_72'], ['A_93', 'a_93', 'A_94', 'a_94'], ['A_73', 'a_73'], ['A_95', 'a_95', 'A_96', 'a_96'], ['A_74', 'a_74'], ['A_97', 'a_97', 'A_98', 'a_98'], ['A_75', 'a_75'], ['A_76', 'a_76', 'A_77', 'a_77'], ['A_78', 'a_78'], ['A_79', 'a_79', 'A_80', 'a_80'], ['A_77', 'a_77'], ['A_81', 'a_81', 'A_82', 'a_82'], ['A_78', 'a_78'], ['A_83', 'a_83', 'A_84', 'a_84'], ['A_79', 'a_79'], ['A_85', 'a_85', 'A_86', 'a_86'], ['A_80', 'a_80'], ['A_87', 'a_87', 'A_88', 'a_88'], ['A_81', 'a_81'], ['A_89', 'a_89', 'A_90', 'a_90'], ['A_82', 'a_82'], ['A_91', 'a_91', 'A_92', 'a_92'], ['A_83', 'a_83'], ['A_93', 'a_93', 'A_94', 'a_94'], ['A_84', 'a_84'], ['A_95', 'a_95', 'A_96', 'a_96'], ['A_85', 'a_85'], ['A_97', 'a_97', 'A_98', 'a_98'], ['A_86', 'a_86'], ['A_87', 'a_87', 'A_88', 'a_88'], ['A_89', 'a_89'], ['A_91', 'a_91', 'A_92', 'a_92'], ['A_90', 'a_90'], ['A_93', 'a_93', 'A_94', 'a_94'], ['A_91', 'a_91'], ['A_95', 'a_95', 'A_96', 'a_96'], ['A_92', 'a_92'], ['A_97', 'a_97', 'A_98', 'a_98'], ['A_93', 'a_93'], ['A_99', 'a_99', 'A_100', 'a_100'], ['A_94', 'a_94'], ['A_95', 'a_95', 'A_96', 'a_96'], ['A_97', 'a_97'], ['A_99', 'a_99', 'A_100', 'a_100'], ['A_98', 'a_98']]
# Crear y evaluar
checker = RiverCrossingChecker(N, k, moves)
result = checker.check()

if result:
    print("✅ La secuencia es válida y todos han cruzado correctamente.")
else:
    print(f"❌ Error en el paso {checker.failed_step + 1}. Elementos conflictivos: {checker.failed_people}")
