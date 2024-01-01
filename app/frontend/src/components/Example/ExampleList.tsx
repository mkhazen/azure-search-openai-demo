import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "What is standard language for Force Majeure in a Wholesale PPA?",
        value: "What is standard language for Force Majeure in a Wholesale PPA?"
    },
    { text: "Who the buyer in the School House PPA", value: "Who the buyer in the School House PPA?" },
    { text: "What is the energy payment Rate in the School House PPA?", value: "What is the energy payment Rate in the School House PPA??" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
