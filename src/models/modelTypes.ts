export type FilteringModel = {
    request(input: string): Promise<boolean>;
}